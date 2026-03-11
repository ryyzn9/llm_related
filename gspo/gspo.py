from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from copy import deepcopy
from datasets import load_dataset
from reward_func import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class GSM8KDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        data = load_dataset(data_path)
        self.data = data['train']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        answer = sample['answer_only']
        prompt = sample['question_zh-cn']
        return {'prompt': prompt, 'answer': answer}


@dataclass
class Samples:
    prompt_response_ids: torch.Tensor
    response_ids:        torch.Tensor
    prompt:              Any
    answer:              Any
    attention_mask:      Optional[torch.LongTensor]
    action_mask:         Optional[torch.BoolTensor]
    num_actions:         Union[int, torch.Tensor]
    response_length:     int


# ════════════════════════════════════════════════════════════════
# DIFFERENCE 1 / 2  ←  CLIP RANGES
#
#  GRPO uses one symmetric range for TOKEN-level ratios:
#       clip_eps = 0.2   →  clip(w_t, 0.8, 1.2)
#
#  GSPO uses asymmetric ranges for SEQUENCE-level ratios.
#  Because s_i is length-normalised (geometric mean of token ratios)
#  it stays near 1.0 naturally, so the ranges are ~500× smaller:
#       clip_left  = 3e-4  →  clip(s_i, 1-3e-4, 1+4e-4)
#       clip_right = 4e-4
# ════════════════════════════════════════════════════════════════

class GSPOArguments:
    output_dir   = './output'
    device       = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr           = 0.000001
    save_steps   = 100
    epoch        = 3
    num_generations  = 4
    max_prompt_length   = 256
    max_generate_length = 256
    reward_weights: List[float] = None
    beta         = 0.0

    # ── GRPO had: clip_eps = 0.2  (token-level, symmetric)
    # ── GSPO:
    clip_left    = 3e-4   # asymmetric left  clip range (paper: 3e-4)
    clip_right   = 4e-4   # asymmetric right clip range (paper: 4e-4)

    gradient_accumulation_steps = 2
    num_iterations = 1
    batch_size   = 1


class GSPOTrainer:
    def __init__(self,
                 model=None,
                 reward_funcs: Union[List[str], List[Callable]] = None,
                 args=None,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset:  Optional[Dataset] = None,
                 tokenizer=None,
                 reward_tokenizers=None):

        self.args = args

        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        self.model = model.to(self.args.device)

        self.ref_model = None
        if self.args.beta != 0.0:
            self.ref_model = deepcopy(model)
            self.ref_model.eval()

        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = self.get_tokenizer(tokenizer)

        if isinstance(reward_funcs, str):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1).to(self.args.device)
        self.reward_funcs = reward_funcs

        if reward_tokenizers is None:
            reward_tokenizers = [None] * len(reward_funcs)
        elif isinstance(reward_tokenizers, str):
            reward_tokenizers = [reward_tokenizers]
        else:
            if len(reward_tokenizers) != len(reward_funcs):
                raise ValueError("Length of reward_tokenizers must equal number of reward_funcs.")
        for i, (reward_tokenizer, reward_func) in enumerate(zip(reward_tokenizers, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_tokenizer is None:
                    reward_tokenizer = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_tokenizer.pad_token_id is None:
                    reward_tokenizer.pad_token = reward_tokenizer.eos_token
                reward_func.config.pad_token_id = reward_tokenizer.pad_token_id
                reward_tokenizers[i] = reward_tokenizer
        self.reward_tokenizers = reward_tokenizers

        self.optimizer   = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_dataset = train_dataset
        self.eval_dataset  = eval_dataset
        self.input_buffer  = [None] * self.args.gradient_accumulation_steps
        self.update_steps  = 0

    def get_tokenizer(self, tokenizer):
        tokenizer.padding_side = "left"
        return tokenizer

    # ── Unchanged from GRPO ──────────────────────────────────────
    def generate_samples(self, inputs):
        samples_list = []
        self.model.eval()
        prompts = [prompt for prompt in inputs['prompt']]
        answers = [None] * len(prompts)
        if 'answer' in inputs:
            answers = [answer for answer in inputs['answer']]

        max_length = self.args.max_generate_length + self.args.max_prompt_length
        for prompt, answer in zip(prompts, answers):
            input_text = self.tokenizer.apply_chat_template(
                [{"role": "system", 'content': SYSTEM_PROMPT},
                 {"role": "user",   'content': prompt}],
                add_generation_prompt=True, tokenize=False
            )
            inputs_tok = self.tokenizer(
                [input_text] * self.args.num_generations,
                padding='max_length', max_length=self.args.max_prompt_length,
                truncation=True, return_tensors='pt'
            )
            prompt_ids = inputs_tok['input_ids']
            with torch.no_grad():
                prompt_response_ids = self.model.generate(
                    **inputs_tok.to(self.args.device),
                    max_new_tokens=self.args.max_generate_length,
                    temperature=0.9, top_p=1, top_k=50
                )
            if prompt_response_ids.size(1) >= max_length:
                prompt_response_ids = prompt_response_ids[:, :max_length]
            else:
                pad = torch.full(
                    (prompt_response_ids.size(0),
                     max_length - prompt_response_ids.size(1)),
                    fill_value=self.tokenizer.pad_token_id,
                    device=prompt_response_ids.device
                )
                prompt_response_ids = torch.cat([prompt_response_ids, pad], dim=1)

            attention_mask = (prompt_response_ids.ne(self.tokenizer.pad_token_id)).long()
            response_ids   = prompt_response_ids[:, prompt_ids.size(1):]
            action_mask    = (
                response_ids.ne(self.tokenizer.eos_token_id) &
                response_ids.ne(self.tokenizer.pad_token_id)
            ).long()

            samples_list.append(Samples(
                prompt_response_ids=prompt_response_ids,
                response_ids=response_ids,
                prompt=prompt,
                answer=answer,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                response_length=action_mask.float().sum(dim=-1)
            ))
        return samples_list

    # ── Unchanged from GRPO ──────────────────────────────────────
    def generate_experiences(self, inputs):
        self.model.eval()
        samples_list = self.generate_samples(inputs)

        batch_prompt_response_ids    = []
        batch_attention_mask         = []
        batch_action_mask            = []
        batch_advantages             = []
        batch_old_action_log_probs   = []
        batch_ref_action_log_probs   = []

        for samples in samples_list:
            prompt_response_ids = samples.prompt_response_ids
            response_ids        = samples.response_ids
            answer              = samples.answer
            attention_mask      = samples.attention_mask
            action_mask         = samples.action_mask
            num_actions         = samples.num_actions
            prompt              = samples.prompt

            batch_prompt_response_ids.append(prompt_response_ids)
            batch_attention_mask.append(attention_mask)
            batch_action_mask.append(action_mask)

            with torch.no_grad():
                old_action_log_probs = self.get_action_log_probs(
                    self.model, prompt_response_ids, attention_mask, num_actions
                )
                batch_old_action_log_probs.append(old_action_log_probs)

                if self.ref_model:
                    ref_action_log_probs = self.get_action_log_probs(
                        self.ref_model, prompt_response_ids, attention_mask, num_actions
                    )
                    batch_ref_action_log_probs.append(ref_action_log_probs)

                rewards_per_func = torch.zeros(
                    len(self.reward_funcs), self.args.num_generations, device=self.args.device
                )
                response_texts       = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                prompt_texts         = [prompt] * len(response_texts)
                prompt_response_texts = [p + r for p, r in zip(prompt_texts, response_texts)]

                for i, (reward_func, reward_tokenizer) in enumerate(
                    zip(self.reward_funcs, self.reward_tokenizers)
                ):
                    if isinstance(reward_func, PreTrainedModel):
                        with torch.inference_mode():
                            reward_model_inputs = reward_tokenizer(
                                prompt_response_texts, return_tensors="pt", padding=True
                            )
                            rewards_per_func[i] = reward_func(
                                **reward_model_inputs.to(self.args.device)
                            ).logits.squeeze(-1)
                    else:
                        answers_list = [answer] * len(prompt_texts)
                        out = reward_func(prompts=prompt_texts, responses=response_texts, answers=answers_list)
                        out = [r if r is not None else torch.nan for r in out]
                        rewards_per_func[i] = torch.tensor(out, dtype=torch.float32, device=self.args.device)

                if not self.args.reward_weights:
                    self.args.reward_weights = [1.0] * len(self.reward_funcs)
                rewards = rewards_per_func * torch.tensor(
                    self.args.reward_weights, dtype=torch.float32,
                    device=rewards_per_func.device
                ).unsqueeze(1)
                rewards = rewards.sum(dim=0)
                print(f'rewards: {rewards}')

                mean_group_rewards = rewards.mean()
                std_group_rewards  = rewards.std()
                advantages = (rewards - mean_group_rewards) / (std_group_rewards + 1e-8)
                batch_advantages.append(advantages)

        return {
            "prompt_response_ids":  torch.cat(batch_prompt_response_ids, dim=0),
            "attention_mask":       torch.cat(batch_attention_mask,       dim=0),
            "action_mask":          torch.cat(batch_action_mask,          dim=0),
            "old_action_log_probs": torch.cat(batch_old_action_log_probs, dim=0),
            "ref_action_log_probs": torch.cat(batch_ref_action_log_probs, dim=0) if self.ref_model else None,
            "advantages":           torch.cat(batch_advantages,           dim=0),
        }

    # ════════════════════════════════════════════════════════════
    # DIFFERENCE 2 / 2  ←  compute_loss  (THE ONLY ALGORITHMIC CHANGE)
    #
    #  GRPO  —  token-level importance ratio + token-level clip:
    #
    #    coef_1 = exp(log_π_θ(y_t) − log_π_old(y_t))          [B, L]
    #    coef_2 = clip(coef_1, 1−0.2, 1+0.2)                   [B, L]
    #    loss   = −mean_t min(coef_1·Â, coef_2·Â)
    #
    #    Problem: coef_1 varies per token → high-variance gradient
    #    signal that accumulates over long sequences → collapse.
    #
    #  GSPO  —  sequence-level importance ratio + sequence-level clip:
    #
    #    s_i = exp( 1/|y| · Σ_t (log_π_θ(y_t) − log_π_old(y_t)) ) [B]
    #        = geometric mean of all per-token ratios
    #    s_i_clip = clip(s_i, 1−3e-4, 1+4e-4)                      [B]
    #    loss = −mean_i  min(s_i·Â_i,  s_i_clip·Â_i)
    #
    #    All tokens in a response get the SAME gradient weight s_i.
    #    One binary clip decision per full response (not per token).
    # ════════════════════════════════════════════════════════════

    def compute_loss(self, model, inputs):
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask      = inputs['attention_mask']
        action_mask         = inputs['action_mask']
        num_actions         = action_mask.size(1)
        advantages          = inputs['advantages']

        action_log_probs = self.get_action_log_probs(
            model, prompt_response_ids, attention_mask, num_actions
        )

        old_action_log_probs = (
            inputs['old_action_log_probs']
            if self.args.num_iterations > 1
            else action_log_probs.detach()
        )

        # ── GSPO: sequence-level importance ratio ────────────────
        # s_i = exp( mean_t [log π_θ(y_t) - log π_old(y_t)] )
        # action_log_probs shape: [B, L]
        log_ratio_masked = (action_log_probs - old_action_log_probs) * action_mask  # [B, L]
        lengths          = action_mask.sum(dim=1).clamp(min=1)                      # [B]
        s_i              = torch.exp(log_ratio_masked.sum(dim=1) / lengths)         # [B]

        # ── GSPO: sequence-level clip (one decision per response) ─
        # GRPO clipped per token at ±0.2; GSPO clips per sequence at ±3e-4/4e-4
        s_i_clipped = torch.clamp(s_i,
                                  1.0 - self.args.clip_left,
                                  1.0 + self.args.clip_right)                       # [B]

        # ── GSPO: sequence-level objective ───────────────────────
        surr1    = s_i         * advantages   # [B]
        surr2    = s_i_clipped * advantages   # [B]
        loss     = -torch.min(surr1, surr2)   # [B]

        # ── Optional KL penalty (unchanged from GRPO) ────────────
        if self.args.beta != 0.0:
            ref_action_log_probs = inputs['ref_action_log_probs']
            # K3 approximation of KL, computed per-token then averaged per sequence
            log_ratio_kl = (ref_action_log_probs - action_log_probs) * action_mask
            k3 = log_ratio_kl.exp() - 1 - log_ratio_kl                 # [B, L]
            kl_per_seq   = k3.sum(dim=1) / lengths                     # [B]
            loss = loss + self.args.beta * kl_per_seq                  # [B]

        return loss.mean()

    # ── Unchanged from GRPO ──────────────────────────────────────
    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        output     = model(input_ids, attention_mask=attention_mask)
        logits     = output.logits
        log_probs  = F.log_softmax(logits[:, :-1, :], dim=-1)
        log_probs_labels = log_probs.gather(
            dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
        )
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
        return action_log_probs

    # ── Unchanged from GRPO ──────────────────────────────────────
    def train_step(self, model, inputs, optimizer, step):
        model.train()
        loss = self.compute_loss(model, inputs)
        loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar("gspo_loss", loss.item(), self.update_steps)
            print(f"step: {self.update_steps}/{self.global_steps}  gspo_loss: {loss.item():.8f}")
        torch.cuda.empty_cache()

    # ── Unchanged from GRPO ──────────────────────────────────────
    def train(self):
        self.global_steps = (
            self.args.num_iterations * self.args.epoch *
            len(self.train_dataset) //
            (self.args.batch_size * self.args.gradient_accumulation_steps)
        )
        for _ in range(self.args.epoch):
            dataloader = DataLoader(
                self.train_dataset, batch_size=self.args.batch_size, shuffle=True
            )
            for idx, batch in enumerate(dataloader):
                inputs = self.generate_experiences(batch)
                self.input_buffer[idx % self.args.gradient_accumulation_steps] = inputs
                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                    for _ in range(self.args.num_iterations):
                        for step, inputs in enumerate(self.input_buffer):
                            self.train_step(self.model, inputs, self.optimizer, step)
                        self.update_steps += 1
                        if self.update_steps % self.args.save_steps == 0:
                            self.model.save_pretrained(
                                self.args.output_dir + f'/checkpoint_{self.update_steps}'
                            )
                            self.tokenizer.save_pretrained(
                                self.args.output_dir + f'/checkpoint_{self.update_steps}'
                            )
                del inputs

    def save_model(self):
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    SYSTEM_PROMPT = """
按照如下格式回答问题：
<think>
你的思考过程
</think>
<answer>
你的回答
</answer>
"""

    args   = GSPOArguments()
    writer = SummaryWriter('./runs')

    tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-1.5B-Instruct')
    model     = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-1.5B-Instruct')

    prompts_dataset = GSM8KDataset(
        '/home/user/wyf/deepseek_learn/gsm8k_chinese', tokenizer
    )

    trainer = GSPOTrainer(
        model=model,
        reward_funcs=[correctness_reward, digit_reward, hard_format_reward, mark_reward],
        args=args,
        train_dataset=prompts_dataset,
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model()
