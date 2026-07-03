"""
CoT推理模型 LoRA 微调脚本

在CoT推理数据上微调Qwen2.5-7B-Instruct，使模型学会：
1. 从自然语言场景中识别约束和实体
2. 逐步推理得出逻辑结论
3. 输出结构化答案JSON

用法:
    python scripts/train_cot_model.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --train_data outputs/cot_train_filtered.json \
        --output_dir checkpoints/cot_model \
        --epochs 3
"""
import json
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Train CoT Reasoning Model with LoRA")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_data", type=str,
                       default="outputs/cot_train_filtered.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints/cot_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("CoT Reasoning Model - LoRA Fine-tuning")
    print("=" * 60)
    print(f"Model:    {args.model_name}")
    print(f"Data:     {args.train_data}")
    print(f"Output:   {args.output_dir}")
    print(f"LoRA:     r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Epochs:   {args.epochs}")
    print(f"Batch:    {args.batch_size} x {args.gradient_accumulation} accumulation")
    print()

    # 检查训练数据
    if not os.path.exists(args.train_data):
        print(f"ERROR: Training data not found: {args.train_data}")
        print("Run generate_cot_data.py first:")
        print(f"  python scripts/generate_cot_data.py --api_key $DEEPSEEK_API_KEY")
        sys.exit(1)

    with open(args.train_data, "r", encoding="utf-8") as f:
        train_examples = json.load(f)
    print(f"Loaded {len(train_examples)} training examples")

    # 按领域统计
    from collections import Counter
    domain_stats = Counter(ex["domain"] for ex in train_examples)
    print(f"Domain distribution: {dict(domain_stats)}")

    # 实际训练（需GPU）
    try:
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer,
            TrainingArguments, Trainer, DataCollatorForSeq2Seq,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        import torch
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install: pip install transformers peft datasets accelerate bitsandbytes")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training on CPU will be very slow.")
        print("This script requires a GPU with >= 16GB VRAM for reasonable training time.")

    # 创建Dataset（只包含ChatML格式的text字段）
    dataset = Dataset.from_list([{"text": ex["text"]} for ex in train_examples])
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Train: {len(dataset['train'])}, Eval: {len(dataset['test'])}")

    # Tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA
    print("Applying LoRA...")
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

    # Training args
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
        run_name="score-cot-reasoning" if args.use_wandb else None,
    )

    # Tokenize
    def tokenize_fn(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )

    print("\nStarting training...")
    trainer.train()

    # Save
    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nModel saved to {final_path}")


if __name__ == "__main__":
    main()
