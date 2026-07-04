"""
约束解析器 LoRA 微调脚本

训练LLM将自然语言场景解析为结构化约束JSON

使用方法:
    python scripts/train_constraint_parser.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --train_data outputs/constraint_annotations.json \
        --output_dir checkpoints/constraint_parser \
        --epochs 3 \
        --batch_size 4
"""
import json
import os
import sys
import argparse
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def build_training_examples(train_data_path: str, output_path: str):
    """
    从训练集构建约束解析的训练样例

    每个训练样例包含：
    - input: 自然语言场景 + 领域类型
    - output: 结构化约束JSON
    """
    with open(train_data_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    examples = []

    for sample in train_data:
        domain = sample["domain"]
        text = sample["text"]
        question = sample.get("question", "")
        language = sample.get("language", "cn")

        # 构建训练样例（此处需要人工标注或LLM生成的约束JSON）
        # 在实际使用前需要先运行约束标注流程
        example = {
            "id": sample["id"],
            "domain": domain,
            "language": language,
            "input": {
                "text": text,
                "question": question,
                "domain": domain,
            },
            "output": {
                "constraints": "TODO: 需要标注"  # 占位
            }
        }
        examples.append(example)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    print(f"Built {len(examples)} training examples → {output_path}")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Train Constraint Parser with LoRA")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Base model name or path")
    parser.add_argument("--train_data", type=str,
                       default="outputs/constraint_annotations.json",
                       help="Training data with constraint annotations")
    parser.add_argument("--output_dir", type=str, default="checkpoints/constraint_parser",
                       help="Output directory for saved model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    args = parser.parse_args()

    print("=" * 60)
    print("Constraint Parser LoRA Fine-tuning")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Train data: {args.train_data}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}")
    print()

    # 检查训练数据是否存在
    if not os.path.exists(args.train_data):
        print(f"⚠️  Training data not found: {args.train_data}")
        print("Please run constraint annotation first:")
        print("  1. python scripts/prepare_constraint_labels.py  # 准备标注数据")
        print("  2. Use LLM to generate constraint annotations")
        print("  3. Place annotations at:", args.train_data)
        return

    # 实际训练代码（需要GPU环境）
    print("Loading model and tokenizer...")

    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForSeq2Seq,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        import torch
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e}")
        print("Please install: pip install transformers peft datasets accelerate bitsandbytes")
        return

    # 加载训练数据
    with open(args.train_data, "r", encoding="utf-8") as f:
        train_examples = json.load(f)

    # 格式化为模型输入
    def format_example(example):
        """将训练样例格式化为模型输入/输出"""
        system_prompt = "You are an expert at extracting structured constraints from commonsense reasoning scenarios. Output valid JSON only."

        domain_prompts = {
            "time": "Extract temporal constraints: events with absolute times and relative time offsets.",
            "space": "Extract spatial constraints: entity positions and spatial relations in a layout.",
            "social": "Extract social constraints: people and their kinship/social relationships.",
            "nature": "Extract natural property constraints: entity attributes and category memberships.",
        }
        domain = example.get("domain", "time")
        domain_instruction = domain_prompts.get(domain.split("+")[0], domain_prompts["time"])

        user_msg = (
            f"## Domain: {domain}\n"
            f"## Task: {domain_instruction}\n\n"
            f"### Scenario:\n{example['input']['text']}\n\n"
            f"### Question:\n{example['input']['question']}\n\n"
            f"### Structured Constraints (JSON):"
        )

        # 使用ChatML格式
        text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{json.dumps(example['output']['constraints'], ensure_ascii=False)}<|im_end|>"
        )
        return {"text": text}

    # 创建HuggingFace Dataset
    dataset = Dataset.from_list(train_examples)
    dataset = dataset.map(format_example)

    # 划分训练/验证集
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        bf16=True,
        report_to="wandb" if args.use_wandb else "none",
        run_name="score-constraint-parser" if args.use_wandb else None,
    )

    # 数据整理器
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    print("\nStarting training...")
    trainer.train()

    # 保存模型
    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✓ Model saved to {final_path}")


if __name__ == "__main__":
    main()
