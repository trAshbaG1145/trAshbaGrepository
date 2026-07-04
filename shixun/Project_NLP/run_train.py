"""
SCoRE2026 CoT 模型训练脚本（云平台一键运行）

用法:
    python run_train.py                          # 完整流程
    python run_train.py --train_only             # 仅训练
    python run_train.py --infer_only             # 仅推理
    python run_train.py --model_name Qwen/Qwen2.5-7B-Instruct

云平台适配:
    AutoDL:    pip install ... && python run_train.py
    Colab:     上传项目文件夹，选择GPU运行时，!python run_train.py
    其他:      确保 CUDA 可用，显存 >= 16GB

输出:
    checkpoints/cot_model/final/   - LoRA 权重
    outputs/submission_cot.json    - 测试集提交文件
"""

import os
import sys
import json
import subprocess
import argparse
import time
from datetime import datetime

# ============================================================
# 配置（根据云平台修改）
# ============================================================

class Config:
    # 模型
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

    # 国内 HuggingFace 镜像（AutoDL 等建议开启）
    HF_MIRROR = "https://hf-mirror.com"
    USE_HF_MIRROR = True  # 云平台设为 True

    # 训练超参
    EPOCHS = 3
    BATCH_SIZE = 4
    GRAD_ACCUM = 4
    LEARNING_RATE = 2e-4
    LORA_R = 16
    LORA_ALPHA = 32
    MAX_LENGTH = 2048

    # 路径
    TRAIN_DATA = "outputs/cot_train_filtered.json"
    OUTPUT_DIR = "checkpoints/cot_model"
    SUBMISSION_DIR = "outputs"

    # 日志
    LOG_FILE = "train.log"


# ============================================================
# 工具函数
# ============================================================

def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(Config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(cmd: str, desc: str = ""):
    """运行shell命令并记录"""
    if desc:
        log(f"--- {desc} ---")
    log(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        log(result.stdout[-2000:])  # 只保留最后2000字符
    if result.stderr and result.returncode != 0:
        log(f"  STDERR: {result.stderr[-1000:]}")
    if result.returncode != 0:
        log(f"  WARNING: command returned {result.returncode}")
    return result


def check_env():
    """检查运行环境"""
    log("=" * 60)
    log("环境检查")
    log("=" * 60)

    import torch
    log(f"Python:     {sys.version.split()[0]}")
    log(f"PyTorch:    {torch.__version__}")
    log(f"CUDA:       {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        log(f"GPU:        {torch.cuda.get_device_name(0)}")
        log(f"VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 14:
            log("WARNING: VRAM < 14GB, training may OOM. Reduce batch_size or max_length.")
    else:
        log("WARNING: CUDA NOT AVAILABLE. Training will be extremely slow on CPU.")
        log("This script REQUIRES a GPU with >= 16GB VRAM.")

    # 检查训练数据
    if not os.path.exists(Config.TRAIN_DATA):
        log(f"ERROR: Training data not found: {Config.TRAIN_DATA}")
        log("Please run generate_cot_data.py first on a machine with API access.")
        sys.exit(1)

    with open(Config.TRAIN_DATA, "r", encoding="utf-8") as f:
        train = json.load(f)
    log(f"Training data: {len(train)} samples")

    from collections import Counter
    log(f"Domain distribution: {dict(Counter(ex['domain'] for ex in train))}")


# ============================================================
# 安装依赖
# ============================================================

def install_deps():
    """安装训练所需依赖"""
    log("=" * 60)
    log("安装依赖")
    log("=" * 60)

    deps = [
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.14.0",
        "accelerate>=0.20.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "sentencepiece",
        "tqdm",
    ]

    for dep in deps:
        run_cmd(f"pip install {dep} -q", f"install {dep}")

    log("依赖安装完成")


# ============================================================
# 训练
# ============================================================

def train():
    """LoRA 微调 Qwen2.5-7B"""
    log("=" * 60)
    log("开始训练")
    log("=" * 60)
    log(f"Model:      {Config.MODEL_NAME}")
    log(f"LoRA:       r={Config.LORA_R}, alpha={Config.LORA_ALPHA}")
    log(f"Epochs:     {Config.EPOCHS}")
    log(f"Batch:      {Config.BATCH_SIZE} x {Config.GRAD_ACCUM} accumulation")
    log(f"Max Length: {Config.MAX_LENGTH}")
    log(f"Output:     {Config.OUTPUT_DIR}")

    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    # ======== 加载数据 ========
    log("Loading training data...")
    with open(Config.TRAIN_DATA, "r", encoding="utf-8") as f:
        train_examples = json.load(f)

    dataset = Dataset.from_list([{"text": ex["text"]} for ex in train_examples])
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    log(f"Train: {len(dataset['train'])}, Eval: {len(dataset['test'])}")

    # ======== 加载模型 ========
    log(f"Loading tokenizer: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"Loading model: {Config.MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    log(f"Model loaded. VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # ======== LoRA ========
    log("Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ======== 训练配置 ========
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACCUM,
        learning_rate=Config.LEARNING_RATE,
        warmup_ratio=0.1,
        logging_steps=50,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        bf16=True,
        report_to="none",
        dataloader_num_workers=2,
    )

    # ======== Tokenize ========
    def tokenize_fn(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=Config.MAX_LENGTH,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # ======== 训练 ========
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )

    log("Starting training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    log(f"Training completed in {elapsed/60:.0f} minutes")

    # ======== 保存 ========
    final_path = os.path.join(Config.OUTPUT_DIR, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    log(f"Model saved to {final_path}")

    return final_path


# ============================================================
# 推理
# ============================================================

def inference(model_path: str):
    """在测试集上推理并生成提交文件"""
    log("=" * 60)
    log("开始推理")
    log("=" * 60)
    log(f"Model path: {model_path}")
    log(f"Base model: {Config.MODEL_NAME}")

    import torch
    import re
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # ======== 加载模型 ========
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    log("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    # ======== 加载测试集 ========
    test_path = os.path.join("data", "SCoRE2026_testset.json")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    log(f"Test samples: {len(test_data)}")

    # ======== CoT Prompt ========
    COT_SYSTEM_PROMPT = """You are an expert at commonsense reasoning problems. Your task is to solve multi-step logical reasoning questions by thinking step by step.

Follow this exact format in your response:

## Reasoning
[Your detailed step-by-step reasoning.]

## Answer
{"answers": ["A", "B"]}

IMPORTANT: Multiple answers may be correct. If the question says "不正确" or "incorrect", select the WRONG option(s). For fill-in-the-blank (____), select the option(s) that fill the blank correctly."""

    def build_prompt(sample):
        options_text = "\n".join(
            f"{l}: {sample['options'][l]}"
            for l in sorted(sample['options'].keys())
        )
        user = (
            f"## Scenario:\n{sample['text']}\n\n"
            f"## Question:\n{sample['question']}\n\n"
            f"## Options:\n{options_text}\n\n"
            f"Think step by step and determine the correct answer(s)."
        )
        return (
            f"<|im_start|>system\n{COT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def extract_answer(response):
        # {"answers": ["A", "B"]}
        for match in re.finditer(r'\{[^}]*"answers"\s*:\s*\[[^\]]*\][^}]*\}', response):
            try:
                obj = json.loads(match.group(0))
                if "answers" in obj and isinstance(obj["answers"], list):
                    return obj["answers"]
            except json.JSONDecodeError:
                pass
        # "answers": ["A", "B"]
        json_match = re.search(r'"answers"\s*:\s*\[(.*?)\]', response, re.DOTALL)
        if json_match:
            answers = re.findall(r'"([A-D])"', json_match.group(1))
            if answers:
                return answers
        # Answer: A, B
        line = re.search(r'(?:answer|答案)[:\s]*([A-D,\s]+)', response, re.IGNORECASE)
        if line:
            return re.findall(r'[A-D]', line.group(1))
        return []

    # ======== 批量推理 ========
    results = []
    for i, sample in enumerate(test_data):
        prompt = build_prompt(sample)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        predicted = extract_answer(response)
        results.append({"id": sample["id"], "answers": predicted})

        if (i + 1) % 100 == 0:
            non_empty = sum(1 for r in results if r["answers"])
            log(f"  [{i+1}/{len(test_data)}] non-empty: {non_empty}")

    # ======== 保存提交 ========
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    output_path = os.path.join(Config.SUBMISSION_DIR, "submission_cot.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    non_empty = sum(1 for r in results if r["answers"])
    log(f"Submission saved to {output_path}")
    log(f"Total: {len(results)}, Non-empty: {non_empty} "
        f"({non_empty/max(len(results),1)*100:.1f}%)")

    from collections import Counter
    answer_counts = Counter()
    for r in results:
        for a in r["answers"]:
            answer_counts[a] += 1
    log(f"Answer distribution: {dict(sorted(answer_counts.items()))}")

    return output_path


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SCoRE2026 CoT Training & Inference (Cloud GPU)")
    parser.add_argument("--train_only", action="store_true", help="仅训练")
    parser.add_argument("--infer_only", action="store_true", help="仅推理")
    parser.add_argument("--model_name", type=str, default=None,
                       help=f"模型名称 (默认: {Config.MODEL_NAME})")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_hf_mirror", action="store_true", help="不使用HF镜像")
    args = parser.parse_args()

    # 应用配置
    if args.model_name:
        Config.MODEL_NAME = args.model_name
    if args.epochs:
        Config.EPOCHS = args.epochs
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.output_dir:
        Config.OUTPUT_DIR = args.output_dir
    if args.no_hf_mirror:
        Config.USE_HF_MIRROR = False

    # HF镜像
    if Config.USE_HF_MIRROR:
        os.environ["HF_ENDPOINT"] = Config.HF_MIRROR
        log(f"Using HF mirror: {Config.HF_MIRROR}")

    log("=" * 60)
    log("SCoRE2026 CoT Training Pipeline")
    log("=" * 60)
    log(f"Time: {datetime.now().isoformat()}")

    # 环境检查
    check_env()

    if args.infer_only:
        model_path = os.path.join(Config.OUTPUT_DIR, "final")
        if not os.path.exists(model_path):
            log(f"ERROR: Model not found at {model_path}")
            log("Run training first: python run_train.py --train_only")
            sys.exit(1)
        inference(model_path)
    elif args.train_only:
        model_path = train()
        log(f"\n{'=' * 60}")
        log(f"Training done! Model: {model_path}")
        log(f"Run inference: python run_train.py --infer_only")
        log(f"{'=' * 60}")
    else:
        # 完整流程
        model_path = train()
        inference(model_path)

        log(f"\n{'=' * 60}")
        log("全部完成!")
        log(f"  模型:   {model_path}")
        log(f"  提交文件: outputs/submission_cot.json")
        log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
