from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel
)
from dataclasses import dataclass
from typing import Callable, List, Optional, Union, Any
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from reward_func import *
import os

# =========================================================
# Environment
# =========================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# =========================================================
# Dataset: GSM8K (Chinese)
# =========================================================
class GSM8KDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = load_dataset(data_path)["train"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "prompt": sample["question_zh-cn"],
            "answer": sample["answer_only"],
        }


# =========================================================
# Grouped samples (DAPO / GRPO setting)
# =========================================================
@dataclass
class Samples:
    prompt_response_ids: torch.Tensor
    response_ids: torch.Tensor
    prompt: Any
    answer: Any
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: int


# =========================================================
# Hyperparameters
# =========================================================
class GRPOArguments:
    output_dir = "./output"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lr = 1e-6
    epoch = 3
    batch_size = 1

    num_generations = 4  # group size G
    max_prompt_length = 256
    max_generate_length = 256

    clip_eps_low = 0.2
    clip_eps_high = 0.28  # Clip-Higher (DAPO)

    gradient_accumulation_steps = 2
    entropy_coef = 0.001  # entropy regularization (stability)


# =========================================================
# Trainer (DAPO-style)
# =========================================================
class GRPOTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        reward_funcs: List[Callable],
        args: GRPOArguments,
        train_dataset: Dataset,
    ):
        self.args = args
        self.model = model.to(args.device)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"

        self.reward_funcs = reward_funcs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.train_dataset = train_dataset

    # =====================================================
    # Generate grouped samples
    # =====================================================
    def generate_samples(self, batch):
        self.model.eval()
        samples = []

        max_len = self.args.max_prompt_length + self.args.max_generate_length

        for prompt, answer in zip(batch["prompt"], batch["answer"]):
            input_text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            tokenized = self.tokenizer(
                [input_text] * self.args.num_generations,
                padding="max_length",
                truncation=True,
                max_length=self.args.max_prompt_length,
                return_tensors="pt",
            )

            with torch.no_grad():
                output_ids = self.model.generate(
                    **tokenized.to(self.args.device),
                    max_new_tokens=self.args.max_generate_length,
                    temperature=0.9,
                    top_p=1.0,
                    top_k=50,
                )

            # pad / truncate
            if output_ids.size(1) < max_len:
                pad = torch.full(
                    (output_ids.size(0), max_len - output_ids.size(1)),
                    self.tokenizer.pad_token_id,
                    device=output_ids.device,
                )
                output_ids = torch.cat([output_ids, pad], dim=1)
            else:
                output_ids = output_ids[:, :max_len]

            attention_mask = (output_ids != self.tokenizer.pad_token_id).long()

            response_ids = output_ids[:, tokenized["input_ids"].size(1) :]

            # EOS-safe action mask (no gradient after EOS)
            eos_mask = torch.cumsum(
                response_ids == self.tokenizer.eos_token_id, dim=1
            )
            action_mask = (eos_mask == 0).long()

            samples.append(
                Samples(
                    prompt_response_ids=output_ids,
                    response_ids=response_ids,
                    prompt=prompt,
                    answer=answer,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    num_actions=action_mask.size(1),
                )
            )

        return samples

    # =====================================================
    # Experience generation (DAPO Eq. 9 & Eq. 11)
    # =====================================================
    def generate_experiences(self, batch):
        samples_list = self.generate_samples(batch)

        buffer = []

        for samples in samples_list:
            with torch.no_grad():
                responses = self.tokenizer.batch_decode(
                    samples.response_ids, skip_special_tokens=True
                )
                prompts = [samples.prompt] * len(responses)
                answers = [samples.answer] * len(responses)

                rewards = []
                for rf in self.reward_funcs:
                    rewards.append(
                        torch.tensor(
                            rf(prompts, responses, answers),
                            device=self.args.device,
                        )
                    )

                rewards = torch.stack(rewards).sum(0)
                rewards = torch.nan_to_num(rewards, nan=0.0)

                # -------------------------------------------------
                # DAPO Eq. (9): Group-relative advantage
                # -------------------------------------------------
                mean = rewards.mean()
                std = rewards.std(unbiased=False).clamp_min(1e-4)
                advantages = (rewards - mean) / std

                # -------------------------------------------------
                # Dynamic Sampling (Eq. 11)
                # -------------------------------------------------
                if advantages.count_nonzero() == 0:
                    continue

                old_log_probs = self.get_action_log_probs(
                    self.model,
                    samples.prompt_response_ids,
                    samples.attention_mask,
                    samples.num_actions,
                )

                buffer.append(
                    {
                        "input_ids": samples.prompt_response_ids,
                        "attention_mask": samples.attention_mask,
                        "action_mask": samples.action_mask,
                        "old_log_probs": old_log_probs,
                        "advantages": advantages,
                    }
                )

        return buffer

    # =====================================================
    # Log-probabilities
    # =====================================================
    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        logits = model(input_ids, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        selected = log_probs.gather(
            -1, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        return selected[:, -num_actions:]

    # =====================================================
    # Loss (DAPO Eq. 8 + Token-level Eq. 12)
    # =====================================================
    def compute_loss(self, batch):
        action_log_probs = self.get_action_log_probs(
            self.model,
            batch["input_ids"],
            batch["attention_mask"],
            batch["action_mask"].size(1),
        )

        # Eq. (6): importance ratio
        ratio = torch.exp(action_log_probs - batch["old_log_probs"])

        # Eq. (8): Clip-Higher surrogate
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.args.clip_eps_low,
            1 + self.args.clip_eps_high,
        )

        advantages = batch["advantages"].unsqueeze(1)

        per_token_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages,
        )

        per_token_loss = per_token_loss * batch["action_mask"]

        # Token-level aggregation (Eq. 12)
        loss = per_token_loss.sum(dim=-1).mean()

        # Entropy regularization (stability)
        entropy = -(action_log_probs.exp() * action_log_probs).sum(-1).mean()
        loss -= self.args.entropy_coef * entropy

        return loss

    # =====================================================
    # Training loop
    # =====================================================
    def train(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
        )

        step = 0
        for epoch in range(self.args.epoch):
            for batch in dataloader:
                experiences = self.generate_experiences(batch)

                for exp in experiences:
                    loss = self.compute_loss(exp)
                    loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                print(f"[epoch {epoch}] step {step} | loss = {loss.item():.6f}")
                step += 1

    def save_model(self):
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    SYSTEM_PROMPT = """
Answer the question in the following format:
<think>
Your reasoning process
</think>
<answer>
Your final answer
</answer>
"""

    args = GRPOArguments()

    tokenizer = AutoTokenizer.from_pretrained(
        "/home/user/Downloads/Qwen2.5-3B-Instruct"
    )
    model = AutoModelForCausalLM.from_pretrained(
        "/home/user/Downloads/Qwen2.5-3B-Instruct"
    )

    dataset = GSM8KDataset(
        "/home/user/wyf/deepseek_learn/gsm8k_chinese",
        tokenizer,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[
            correctness_reward,
            digit_reward,
            hard_format_reward,
            mark_reward,
        ],
        args=args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model()
