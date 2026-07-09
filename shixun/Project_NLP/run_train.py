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

    # ModelScope（国内更稳定，免翻墙）
    USE_MODELSCOPE = False  # 运行时可通过 --modelscope 开启

    # 训练超参（32GB V100: batch=2/2048 安全）
    EPOCHS = 3
    BATCH_SIZE = 2       # 降到 2 保证 max_length=2048 不 OOM
    GRAD_ACCUM = 8       # 有效 batch = 2×8 = 16（不变）
    LEARNING_RATE = 2e-4
    LORA_R = 16
    LORA_ALPHA = 32
    MAX_LENGTH = 2048

    # 路径
    TRAIN_DATA = "outputs/cot_train_filtered.json"
    OUTPUT_DIR = "checkpoints/cot_model"
    SUBMISSION_DIR = "outputs"
    LOG_DIR = "logs"


# ============================================================
# 工具函数
# ============================================================

def log(msg: str):
    """记录带时间戳的消息（Tee 已捕获到文件，这里只负责格式化）"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


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
        # 诊断 CUDA 不可用的原因
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                log("ERROR: GPU detected by nvidia-smi but CUDA is unavailable in PyTorch.")
                log("This usually means PyTorch was compiled for a different CUDA version.")
                log("Fix: install a compatible PyTorch version. For Tesla V100 (CUDA 12.0):")
                log("  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121")
                log("For other GPUs, check https://pytorch.org/get-started/locally/")
            else:
                log("WARNING: No GPU detected. Training will be extremely slow on CPU.")
        except Exception:
            log("WARNING: CUDA NOT AVAILABLE. Training will be extremely slow on CPU.")
        log("This script REQUIRES a GPU with >= 16GB VRAM.")
        log("Aborting — please fix the CUDA/PyTorch mismatch above and re-run.")
        sys.exit(1)

    # 检查训练数据（推理模式不需要）
    if os.path.exists(Config.TRAIN_DATA):
        with open(Config.TRAIN_DATA, "r", encoding="utf-8") as f:
            train = json.load(f)
        log(f"Training data: {len(train)} samples")
        from collections import Counter
        log(f"Domain distribution: {dict(Counter(ex['domain'] for ex in train))}")
    else:
        log("Training data not found (--infer_only mode, skipping)")


# ============================================================
# 安装依赖
# ============================================================

def install_deps():
    """安装训练所需依赖"""
    log("=" * 60)
    log("安装依赖")
    log("=" * 60)

    deps = [
        "transformers>=4.40.0",
        "datasets>=2.14.0",
        "accelerate>=0.20.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "sentencepiece",
        "tqdm",
    ]

    # 移除可能冲突的 torch 生态包（文本模型不需要 vision/audio）
    # 云平台常预装 CUDA 版本不匹配的 torchvision/torchaudio，导致 import 时报 libcudart.so 错误
    log("Removing torchvision/torchaudio if present (not needed for Qwen2.5)...")
    subprocess.run("pip uninstall torchvision torchaudio -y 2>/dev/null",
                   shell=True, capture_output=True, text=True)

    # 一次性安装所有依赖
    dep_str = " ".join(f'"{d}"' for d in deps)
    log("Installing dependencies (this may take a few minutes)...")
    result = subprocess.run(
        f"pip install {dep_str}", shell=True,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log(f"ERROR: pip install failed (code {result.returncode})")
        log(f"STDERR: {result.stderr[-2000:]}")
        log("Try: pip install transformers peft datasets accelerate bitsandbytes sentencepiece tqdm")
    else:
        for line in result.stdout.split('\n'):
            if 'Successfully installed' in line or 'already satisfied' in line:
                log(line.strip())

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
    if torch.cuda.is_available():
        log(f"Model loaded. VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    else:
        log("Model loaded (CPU)")

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

    # Gradient checkpointing: 用计算换显存（节省 ~40% 激活内存）
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    log("Gradient checkpointing enabled")

    # 减少显存碎片
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # 防止 DataLoader 多进程与 tokenizers 死锁
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ======== 训练配置 ========
    if not torch.cuda.is_available():
        log("ERROR: GPU required for training. Aborting.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    use_bf16 = torch.cuda.is_bf16_supported()
    if use_bf16:
        log(f"Using bf16 precision (GPU: {gpu_name})")
    else:
        log(f"WARNING: bf16 not supported on {gpu_name}, using fp16")

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
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="none",
        dataloader_num_workers=0,  # 避免多进程 fork 死锁（特别是旧内核）
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
    from tqdm import tqdm

    # ======== 加载模型 ========
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # V100 不支持 bf16，用 fp16（Tensor Core 原生加速 ~2x）
    log("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).cuda()
    # 禁用 sampling 默认值，消除 generate() 的 warning
    base_model.generation_config.temperature = None
    base_model.generation_config.top_p = None
    base_model.generation_config.top_k = None

    log("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, model_path)
    # 不 merge！merge 会破坏输出格式，导致 90%+ 空答案
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    # ======== 加载测试集 ========
    test_path = os.path.join("data", "SCoRE2026_testset.json")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    log(f"Test samples: {len(test_data)}")

    # ======== CoT Prompt（必须与 generate_cot_data.py 完全一致）========
    COT_SYSTEM_PROMPT = """You are an expert at commonsense reasoning problems. Your task is to solve multi-step logical reasoning questions by thinking step by step.

Follow this exact format in your response:

## Reasoning
[Your detailed step-by-step reasoning. Break down the problem:
1. Identify all entities and their known properties/positions/relationships
2. List all constraints stated in the text
3. Reason through the constraints step by step
4. For each option, check whether it is consistent with your conclusions]

## Answer
{"answers": ["A", "B"]}

IMPORTANT:
- Multiple answers may be correct. List ALL correct options.
- If the question says "不正确" or "incorrect", select the WRONG option(s).
- For fill-in-the-blank (____), select the option(s) that make the statement true.
- Think carefully about every constraint before drawing conclusions."""

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
        # 策略1: JSON 对象 {"answers": ["A", "B"]}
        for match in re.finditer(r'\{[^}]*"answers"\s*:\s*\[[^\]]*\][^}]*\}', response):
            try:
                obj = json.loads(match.group(0))
                if "answers" in obj and isinstance(obj["answers"], list):
                    return obj["answers"]
            except json.JSONDecodeError:
                pass
        # 策略2: "answers": ["A", "B"] 片段匹配
        json_match = re.search(r'"answers"\s*:\s*\[(.*?)\]', response, re.DOTALL)
        if json_match:
            answers = re.findall(r'"([A-D])"', json_match.group(1))
            if answers:
                return answers
        # 策略3: Answer: A, B 纯文本
        line = re.search(r'(?:answer|答案)[:\s]*([A-D,\s]+)', response, re.IGNORECASE)
        if line:
            return re.findall(r'[A-D]', line.group(1))
        # 策略4: 从尾部搜索 JSON（容错）
        tail = response[-500:]
        for match in re.finditer(r'\{[^}]*"answers"\s*:\s*\[[^\]]*\][^}]*\}', tail):
            try:
                obj = json.loads(match.group(0))
                if "answers" in obj and isinstance(obj["answers"], list):
                    return obj["answers"]
            except json.JSONDecodeError:
                pass
        return []

    # ======== 批量推理（支持断点续传）========
    # 加载已有进度
    checkpoint_path = os.path.join(Config.SUBMISSION_DIR, ".infer_checkpoint.json")
    completed_ids = set()
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        completed_ids = {r["id"] for r in results}
        log(f"Resumed {len(results)} completed predictions")

    pbar = tqdm(total=len(test_data), initial=len(completed_ids),
                desc="Inference", unit="samples")
    for i, sample in enumerate(test_data):
        if sample["id"] in completed_ids:
            continue

        prompt = build_prompt(sample)
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        predicted = extract_answer(response)
        results.append({"id": sample["id"], "answers": predicted})
        pbar.update(1)

        # 每 100 条保存检查点
        if (len(results) - len(completed_ids)) % 100 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    pbar.close()

    # ======== 保存提交 ========
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    output_path = os.path.join(Config.SUBMISSION_DIR, "submission_cot.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 清理检查点
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

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

class Tee:
    """同时输出到终端和日志文件，捕获所有 stdout/stderr"""
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.file = open(log_path, "a", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def fileno(self):
        return self.stdout.fileno()

    def close(self):
        pass  # 不关闭 stdout/stderr 本身


def setup_logging():
    """初始化全量日志记录（之后所有终端输出都会被捕获到文件）"""
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(Config.LOG_DIR, f"train_{timestamp}.log")
    tee = Tee(log_path)
    sys.stdout = tee  # type: ignore
    sys.stderr = tee  # type: ignore
    return log_path


def _download_from_modelscope(model_name: str) -> str:
    """从 ModelScope 下载模型，返回本地路径"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        log("Installing modelscope...")
        subprocess.run("pip install modelscope", shell=True, capture_output=True)
        from modelscope import snapshot_download

    log(f"Downloading {model_name} from ModelScope...")
    local_dir = snapshot_download(model_name, cache_dir="models")
    log(f"Model downloaded to: {local_dir}")
    return local_dir


def main():
    # 初始化全量日志（最早执行，确保所有输出被捕获）
    log_path = setup_logging()
    print(f"Log file: {log_path}")

    # 解析 --install
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--install", action="store_true")
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.install:
        install_deps()

    # 完整参数解析
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
    parser.add_argument("--modelscope", action="store_true", help="使用 ModelScope 下载模型（国内更快）")
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
    if args.modelscope:
        Config.USE_MODELSCOPE = True

    # 模型加载方式
    if Config.USE_MODELSCOPE:
        log("Using ModelScope for model download...")
        Config.MODEL_NAME = _download_from_modelscope(Config.MODEL_NAME)
        Config.USE_HF_MIRROR = False  # ModelScope 不需要 HF
    elif Config.USE_HF_MIRROR:
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
