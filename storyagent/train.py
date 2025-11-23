# train_qwen_lora.py
import os
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"  # 小模型
DATA_PATH = "data/mystery_town_qwen_train.jsonl"
OUTPUT_DIR = "qwen0_5b_mystery_lora"


@dataclass
class QwenDetectiveExample:
    prompt: str
    answer: str


def load_train_dataset() -> Any:
    """
    从 jsonl 加载数据，其中每行包含 'prompt' 和 'answer'
    """
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    return dataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    # Qwen1.5 一般有 chat 模式，这里简单用纯文本格式来做 SFT：
    EOS = tokenizer.eos_token or "<|endoftext|>"

    raw_dataset = load_train_dataset()

    max_length = 512  # 对话很长可以适当裁剪，先保守一些

    def format_example(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        把一条样本变成：<prompt>\n答案：<answer>
        再做统一的 tokenization。
        """
        prompt = example["prompt"]
        answer = example["answer"].strip()
        text = f"{prompt}\n答案：{answer}{EOS}"

        tokenized = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
        )
        # 语言模型监督微调：labels 就是 input_ids 的 copy
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)

    # 加载 Qwen 模型（0.5B），不做量化也能跑，显存压力不大
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    # 配置 LoRA
    # 注意：target_modules 需要根据实际模型结构调整，这里给一个典型示例，
    # 如果报错，可以通过打印 model.named_modules() 来进一步筛选。
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "w1", "w2", "w3"],  # 示例，必要时需调整
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    trainer.train()

    # 保存 LoRA 权重 + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"模型已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
