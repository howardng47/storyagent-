import os
from dataclasses import dataclass

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ========= 配置 =========

MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"  # 千问 0.5B 聊天模型
DATA_PATH = "data/mystery_town_qwen_train.jsonl"
OUTPUT_DIR = "qwen_mysterytown_lora"

MAX_SEQ_LEN = 512


def main():
    # 1. 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",          # 自动放到 GPU
        torch_dtype="auto",
    )

    # 2. 配置 LoRA（只调注意力矩阵 q/k/v，比较省显存）
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 加载数据集
    dataset = load_dataset("json", data_files=DATA_PATH)["train"]

    def format_example(example):
        # 把 prompt + 答案串成一条文本，让模型学习“看到推理过程 -> 输出凶手ID”
        text = example["prompt"].strip()
        answer = example["answer"].strip()

        # 简单格式：问答式
        full_text = (
            text
            + "\n答案："
            + answer
            + tokenizer.eos_token
        )

        tokenized = tokenizer(
            full_text,
            max_length=MAX_SEQ_LEN,
            truncation=True,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_ds = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 4. 训练参数（可以根据显存调整 batch_size 和 gradient_accumulation_steps）
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    trainer.train()

    # 5. 只保存 LoRA 权重（体积很小）
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
