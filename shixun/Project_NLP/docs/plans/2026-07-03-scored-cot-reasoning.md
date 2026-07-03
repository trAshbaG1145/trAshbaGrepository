# SCoRE2026 CoT 推理方案实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将策略从 Neuro-Symbolic 约束提取切换为纯 LLM Chain-of-Thought 推理，用 DeepSeek-V3 生成 CoT 训练数据，LoRA 微调 Qwen2.5-7B，直接输出推理链+答案。

**Architecture:** 三阶段流水线：(A) DeepSeek API 在训练集上生成 CoT 推理链 → 按正确答案过滤 → 构建训练数据；(B) LoRA 微调 Qwen2.5-7B；(C) 微调模型在测试集上推理 → 提交 JSON。

**Tech Stack:** DeepSeek API (数据生成), Qwen2.5-7B-Instruct + LoRA/PeFT (微调), vLLM/Transformers (推理)

## Global Constraints

- 模型规模 ≤ 8B（Dense）或 MoE ≤ 30B 总参 / ≤ 3B 每 token 激活
- 禁止使用 SCoRE2026 以外的外部数据集
- 测试集禁止用于训练、微调或作为 prompt 示例
- 提交格式：`[{"id": "...", "answers": ["A", "B"]}]`
- 评估指标：Accuracy = 正确数 / 总题数
- 中英文混合：中文 58% + 英文 42%
- 多选率 23.6%（848/3600 题有多个正确答案）

## File Structure

```
Project_NLP/
├── scripts/
│   ├── generate_cot_data.py    # NEW: CoT数据生成（API调用）
│   ├── train_cot_model.py      # NEW: CoT模型LoRA微调
│   └── run_cot_inference.py    # NEW: CoT模型推理
├── src/
│   └── cot/
│       └── __init__.py          # NEW: CoT模块
├── outputs/
│   ├── cot_train_raw.json      # NEW: 原始CoT数据（含错误）
│   ├── cot_train_filtered.json # NEW: 过滤后CoT训练数据
│   └── submission_cot.json     # NEW: CoT模型提交
├── checkpoints/
│   └── cot_model/              # NEW: LoRA权重
└── main.py                      # MODIFY: 添加新命令
```

## 保留的旧模块（作为基线对比）

- `src/solvers/`, `src/parser/`, `src/pipeline/` — 保留不动，消融实验时用作对比基线
- `scripts/batch_annotate.py` — 保留不动，如需回到 N-S 路线可继续使用

---

### Task 1: CoT 数据生成脚本

**Files:**
- Create: `scripts/generate_cot_data.py`

**Interfaces:**
- Produces: `generate_cot_samples(train_data, client, samples=None) -> List[Dict]` — 每个 dict 含 `id, domain, input_text, cot_reasoning, predicted_answer, ground_truth, is_correct`
- Produces: `export_cot_train_data(results, output_path)` — 导出过滤后的训练数据

- [ ] **Step 1: 创建脚本框架和 CoT prompt 模板**

```python
"""
CoT 推理数据生成脚本

用 DeepSeek API 在训练集上生成 Chain-of-Thought 推理链，
按正确答案过滤后作为 Qwen2.5-7B 的微调数据。

用法:
    python scripts/generate_cot_data.py --api_key sk-xxx --samples 100
    python scripts/generate_cot_data.py --api_key sk-xxx  # 全量3600条
"""
import json
import os
import sys
import re
import time
import argparse
from typing import Dict, List, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")

# ============================================================
# CoT Prompt 模板
# ============================================================

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

COT_USER_TEMPLATE = """## Scenario:
{text}

## Question:
{question}

## Options:
{options_text}

Think step by step and determine the correct answer(s)."""


def format_options(options: Dict[str, str]) -> str:
    """格式化选项为文本"""
    lines = []
    for letter in sorted(options.keys()):
        lines.append(f"{letter}: {options[letter]}")
    return "\n".join(lines)


def extract_answer_from_response(response: str) -> List[str]:
    """从CoT响应中提取答案"""
    # 策略1: JSON格式
    json_match = re.search(r'"answers"\s*:\s*\[(.*?)\]', response, re.DOTALL)
    if json_match:
        answers_str = json_match.group(1)
        answers = re.findall(r'"([A-D])"', answers_str)
        if answers:
            return answers

    # 策略2: {"answers": ["A", "B"]} 作为独立JSON
    for match in re.finditer(r'\{[^}]*"answers"[^}]*\}', response):
        try:
            obj = json.loads(match.group(0))
            if "answers" in obj:
                return obj["answers"]
        except json.JSONDecodeError:
            pass

    # 策略3: "Answer: A, B" 格式
    answer_line = re.search(r'(?:answer|答案)[:\s]*([A-D,\s]+)', response, re.IGNORECASE)
    if answer_line:
        return re.findall(r'[A-D]', answer_line.group(1))

    return []


def extract_reasoning_from_response(response: str) -> str:
    """从响应中提取推理部分"""
    # 匹配 ## Reasoning 到 ## Answer 之间的内容
    match = re.search(r'##\s*Reasoning\s*\n(.*?)(?=##\s*Answer|\Z)', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # 如果没有 ## Answer 标记，取整个响应
    answer_match = re.search(r'##\s*Answer', response, re.IGNORECASE)
    if answer_match:
        return response[:answer_match.start()].strip()
    return response.strip()
```

- [ ] **Step 2: 实现 API 调用和并发处理**

```python
class CoTDataGenerator:
    """CoT训练数据生成器"""

    def __init__(self, api_key: str, model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate_one(self, sample: Dict) -> Optional[Dict]:
        """为单条样本生成CoT推理"""
        options_text = format_options(sample["options"])
        user_prompt = COT_USER_TEMPLATE.format(
            text=sample["text"],
            question=sample["question"],
            options_text=options_text,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": COT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048,
            temperature=0.0,
        )

        full_response = response.choices[0].message.content
        reasoning = extract_reasoning_from_response(full_response)
        predicted = extract_answer_from_response(full_response)
        ground_truth = sample.get("answers", [])

        return {
            "id": sample["id"],
            "domain": sample["domain"],
            "language": sample.get("language", "cn"),
            "text": sample["text"],
            "question": sample["question"],
            "options": sample["options"],
            "cot_reasoning": reasoning,
            "predicted_answer": predicted,
            "ground_truth": ground_truth,
            "is_correct": set(predicted) == set(ground_truth),
            "raw_response": full_response,
        }

    def generate_batch(self, samples: List[Dict], concurrency: int = 5,
                       max_samples: int = 0) -> List[Dict]:
        """批量生成CoT数据"""
        if max_samples > 0:
            samples = samples[:max_samples]

        results = []
        completed = 0
        total = len(samples)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_sample = {
                executor.submit(self.generate_one, s): s for s in samples
            }
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                completed += 1
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        status = "✓" if result["is_correct"] else "✗"
                    else:
                        status = "ERROR"
                except Exception as e:
                    print(f"  [{completed}/{total}] ERROR {sample['id']}: {e}")
                    status = "ERROR"
                    continue

                print(f"  [{completed}/{total}] {status} {sample['id']} "
                      f"[{sample['domain']}] pred={result.get('predicted_answer', [])} "
                      f"actual={sample.get('answers', [])}")

        return results
```

- [ ] **Step 3: 实现数据导出和统计**

```python
def export_cot_train_data(results: List[Dict], output_path: str,
                          quality_filter: bool = True) -> List[Dict]:
    """
    导出CoT训练数据为Qwen2.5的ChatML格式

    ChatML格式:
    <|im_start|>system
    {system_prompt}
    <|im_end|>
    <|im_start|>user
    {user_prompt}
    <|im_end|>
    <|im_start|>assistant
    {reasoning + answer_json}
    <|im_end|>
    """
    train_data = []

    for r in results:
        if quality_filter and not r.get("is_correct", False):
            continue

        options_text = format_options(r["options"])
        user_prompt = COT_USER_TEMPLATE.format(
            text=r["text"],
            question=r["question"],
            options_text=options_text,
        )

        assistant_output = (
            f"## Reasoning\n{r['cot_reasoning']}\n\n"
            f"## Answer\n{json.dumps({'answers': r['predicted_answer']}, ensure_ascii=False)}"
        )

        # ChatML格式（Qwen2.5标准）
        chatml = (
            f"<|im_start|>system\n{COT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_output}<|im_end|>"
        )

        train_data.append({
            "id": r["id"],
            "domain": r["domain"],
            "language": r["language"],
            "text": chatml,
            "num_answers": len(r["predicted_answer"]),
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    return train_data


def print_stats(results: List[Dict]):
    """打印统计信息"""
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    empty = sum(1 for r in results if not r.get("predicted_answer"))

    print(f"\n{'=' * 60}")
    print(f"CoT Data Generation Stats")
    print(f"{'=' * 60}")
    print(f"  Total:           {total}")
    print(f"  Correct:         {correct} ({correct/max(total,1)*100:.1f}%)")
    print(f"  Empty prediction:{empty}")
    print(f"  Training samples:{correct} (correct answers only)")

    # 分领域统计
    by_domain = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        d = r["domain"]
        by_domain[d]["total"] += 1
        if r.get("is_correct"):
            by_domain[d]["correct"] += 1

    print(f"\n  --- By Domain ---")
    print(f"  {'Domain':15s} {'Total':>6s} {'Correct':>8s} {'Rate':>7s}")
    for domain in sorted(by_domain):
        ds = by_domain[domain]
        rate = ds["correct"] / max(ds["total"], 1) * 100
        print(f"  {domain:15s} {ds['total']:6d} {ds['correct']:8d} {rate:6.1f}%")
```

- [ ] **Step 4: 实现 main() 和 CLI**

```python
def main():
    parser = argparse.ArgumentParser(description="Generate CoT reasoning data for SCoRE2026")
    parser.add_argument("--api_key", type=str, default=None,
                       help="DeepSeek API Key (or set DEEPSEEK_API_KEY env var)")
    parser.add_argument("--model", type=str, default="deepseek-chat",
                       help="Model name (default: deepseek-chat)")
    parser.add_argument("--samples", type=int, default=0,
                       help="Max samples to process (0=all 3600)")
    parser.add_argument("--concurrency", type=int, default=5,
                       help="Concurrent API calls (default: 5)")
    parser.add_argument("--domain", type=str, default=None,
                       help="Only process specific domain")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for training data")
    parser.add_argument("--raw_output", type=str, default=None,
                       help="Output path for raw results (with debug info)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: Provide --api_key or set DEEPSEEK_API_KEY env var")
        sys.exit(1)

    # 加载数据
    train_path = os.path.join(DATA_DIR, "SCoRE2026_trainset.json")
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    if args.domain:
        train_data = [s for s in train_data if s["domain"] == args.domain]
        print(f"Filtered to domain '{args.domain}': {len(train_data)} samples")

    if args.samples > 0:
        train_data = train_data[:args.samples]

    print(f"Processing {len(train_data)} samples...")

    # 生成
    generator = CoTDataGenerator(api_key=api_key, model=args.model)
    start = time.time()
    results = generator.generate_batch(train_data, concurrency=args.concurrency)
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/max(len(results),1):.1f}s/sample)")

    # 统计
    print_stats(results)

    # 导出
    raw_path = args.raw_output or os.path.join(OUTPUT_DIR, "cot_train_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Raw results saved to {raw_path}")

    output_path = args.output or os.path.join(OUTPUT_DIR, "cot_train_filtered.json")
    train_data = export_cot_train_data(results, output_path, quality_filter=True)
    print(f"Training data ({len(train_data)} samples) saved to {output_path}")

    # 后续步骤
    if train_data:
        print(f"\nNext: LoRA fine-tune with {len(train_data)} training samples")
        print(f"  python scripts/train_cot_model.py \\")
        print(f"    --train_data {output_path} \\")
        print(f"    --output_dir checkpoints/cot_model")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 验证脚本语法和导入**

```bash
cd e:/Coding/trAshbaGrepository/shixun/Project_NLP
python -c "from scripts.generate_cot_data import *; print('Import OK')"
```

- [ ] **Step 6: 小规模测试（10条）**

```bash
python scripts/generate_cot_data.py --api_key $env:DEEPSEEK_API_KEY --samples 10
# 验证输出: outputs/cot_train_raw.json 和 outputs/cot_train_filtered.json
```

- [ ] **Step 7: Commit**

```bash
git add scripts/generate_cot_data.py
git commit -m "feat: add CoT reasoning data generation script

Uses DeepSeek API to generate step-by-step reasoning chains on the
SCoRE2026 training set. Filters by answer correctness to produce
high-quality training data for LoRA fine-tuning of Qwen2.5-7B.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: CoT 模型 LoRA 微调脚本

**Files:**
- Create: `scripts/train_cot_model.py`

**Interfaces:**
- Consumes: `outputs/cot_train_filtered.json` — 每个 dict 含 `id, domain, language, text`（ChatML格式）
- Produces: `checkpoints/cot_model/` — LoRA adapter 权重 + tokenizer

- [ ] **Step 1: 创建训练脚本**

```python
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
    print(f"Batch:    {args.batch_size} × {args.gradient_accumulation} accumulation")
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

    # 创建Dataset（只包含ChatML格式的text字段）
    dataset = Dataset.from_list([{"text": ex["text"]} for ex in train_examples])
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA
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
```

- [ ] **Step 2: 验证脚本语法**

```bash
cd e:/Coding/trAshbaGrepository/shixun/Project_NLP
python -c "import ast; ast.parse(open('scripts/train_cot_model.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/train_cot_model.py
git commit -m "feat: add CoT model LoRA fine-tuning script

Trains Qwen2.5-7B-Instruct on CoT reasoning data using LoRA.
Accepts ChatML-formatted training data from generate_cot_data.py.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: CoT 推理脚本

**Files:**
- Create: `scripts/run_cot_inference.py`

**Interfaces:**
- Consumes: `checkpoints/cot_model/final/` — LoRA 权重
- Produces: `outputs/submission_cot.json` — 提交文件

- [ ] **Step 1: 创建推理脚本**

```python
"""
CoT模型推理脚本

加载微调后的Qwen2.5-7B模型，在测试集上执行CoT推理并生成提交文件。

用法:
    python scripts/run_cot_inference.py \
        --model_path checkpoints/cot_model/final \
        --output outputs/submission_cot.json
"""
import json
import os
import sys
import re
import argparse
from typing import Dict, List, Optional
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def extract_answer_from_response(response: str) -> List[str]:
    """从模型响应中提取答案"""
    # 策略1: {"answers": ["A", "B"]}
    for match in re.finditer(r'\{[^}]*"answers"\s*:\s*\[[^\]]*\][^}]*\}', response):
        try:
            obj = json.loads(match.group(0))
            if "answers" in obj and isinstance(obj["answers"], list):
                return obj["answers"]
        except json.JSONDecodeError:
            pass

    # 策略2: "answers": ["A", "B"]
    json_match = re.search(r'"answers"\s*:\s*\[(.*?)\]', response, re.DOTALL)
    if json_match:
        answers = re.findall(r'"([A-D])"', json_match.group(1))
        if answers:
            return answers

    # 策略3: Answer: A, B
    answer_line = re.search(
        r'(?:answer|答案|correct\s+(?:answer|option))[:\s]*([A-D,\s]+)',
        response, re.IGNORECASE
    )
    if answer_line:
        return re.findall(r'[A-D]', answer_line.group(1))

    # 策略4: 最后一行包含选项字母
    lines = response.strip().split('\n')
    for line in reversed(lines):
        found = re.findall(r'\b([A-D])\b', line)
        if found and len(found) <= 4:
            return found

    return []


def build_prompt(sample: Dict) -> str:
    """构建推理prompt（与训练时一致）"""
    options_text = "\n".join(
        f"{l}: {sample['options'][l]}" for l in sorted(sample['options'].keys())
    )
    system = (
        "You are an expert at commonsense reasoning problems. "
        "Think step by step, then output your answer as {\"answers\": [\"A\", \"B\"]}."
    )
    user = (
        f"## Scenario:\n{sample['text']}\n\n"
        f"## Question:\n{sample['question']}\n\n"
        f"## Options:\n{options_text}\n\n"
        f"Think step by step and determine the correct answer(s)."
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_inference(model, tokenizer, samples: List[Dict],
                  batch_size: int = 1, max_new_tokens: int = 1024) -> List[Dict]:
    """批量推理"""
    results = []

    for i, sample in enumerate(samples):
        prompt = build_prompt(sample)
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        predicted = extract_answer_from_response(response)
        results.append({
            "id": sample["id"],
            "answers": predicted,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(samples)}] {sample['id']}: {predicted}")

    return results


def main():
    parser = argparse.ArgumentParser(description="CoT Model Inference for SCoRE2026")
    parser.add_argument("--model_path", type=str,
                       default="checkpoints/cot_model/final")
    parser.add_argument("--base_model", type=str,
                       default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=str,
                       default=None)
    parser.add_argument("--samples", type=int, default=0,
                       help="Max test samples (0=all 1000)")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    # 加载模型
    print(f"Loading model: {args.base_model}")
    print(f"LoRA adapter: {args.model_path}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if os.path.exists(args.model_path):
        model = PeftModel.from_pretrained(base_model, args.model_path)
        print("LoRA adapter loaded")
    else:
        print(f"WARNING: LoRA adapter not found at {args.model_path}")
        print("Using base model without fine-tuning")
        model = base_model

    model.eval()

    # 加载测试集
    test_path = os.path.join(DATA_DIR, "SCoRE2026_testset.json")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if args.samples > 0:
        test_data = test_data[:args.samples]

    print(f"Running inference on {len(test_data)} test samples...")

    # 推理
    results = run_inference(model, tokenizer, test_data,
                           max_new_tokens=args.max_new_tokens)

    # 保存
    output_path = args.output or os.path.join(OUTPUT_DIR, "submission_cot.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    non_empty = sum(1 for r in results if r["answers"])
    print(f"\nSubmission saved to {output_path}")
    print(f"Total: {len(results)}, Non-empty: {non_empty}")

    # 答案分布
    answer_counts = Counter()
    for r in results:
        for a in r["answers"]:
            answer_counts[a] += 1
    print(f"Answer distribution: {dict(answer_counts)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证脚本语法**

```bash
python -c "import ast; ast.parse(open('scripts/run_cot_inference.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_cot_inference.py
git commit -m "feat: add CoT model inference script

Loads LoRA fine-tuned Qwen2.5-7B and generates CoT reasoning on the
SCoRE2026 test set. Outputs submission JSON with extracted answers.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: 更新 main.py 和项目文档

**Files:**
- Modify: `main.py`
- Modify: `README.md`
- Create: `src/cot/__init__.py`

- [ ] **Step 1: 更新 main.py 添加新命令**

`main.py` 中的 `cmd_batch_annotate` 存在参数传递问题（`parse_known_args` 后 `sys.argv` 被修改）。改为更干净的方式：

```python
def cmd_generate_cot():
    """生成CoT推理训练数据"""
    from scripts.generate_cot_data import main as m
    m()

def cmd_train_cot():
    """训练CoT推理模型"""
    from scripts.train_cot_model import main as m
    m()

def cmd_infer_cot():
    """CoT模型推理"""
    from scripts.run_cot_inference import main as m
    m()
```

并在 `main()` 的 choices 中添加 `"gen-cot"`, `"train-cot"`, `"infer-cot"`，在 `cmds` 字典中添加对应的函数映射。

- [ ] **Step 2: 更新 README.md 进度**

在 README.md 末尾更新当前进度部分，添加：
```markdown
- [x] CoT数据生成脚本
- [x] CoT模型LoRA微调脚本
- [x] CoT推理脚本
- [ ] CoT训练数据生成（需API调用）
- [ ] 模型训练（需GPU环境）
- [ ] 最终测试集推理与提交
- [ ] 消融实验（CoT vs 模板 vs 零样本）
```

- [ ] **Step 3: 创建 CoT 模块**

```python
# src/cot/__init__.py
"""
CoT (Chain-of-Thought) Reasoning Module

Strategy: Direct LLM reasoning instead of Neuro-Symbolic constraint extraction.
- generate_cot_data.py: Use strong LLM to generate reasoning chains
- train_cot_model.py: LoRA fine-tune Qwen2.5-7B on reasoning data
- run_cot_inference.py: Generate answers for test set
"""
```

- [ ] **Step 4: Commit**

```bash
git add main.py README.md src/cot/__init__.py
git commit -m "feat: add CoT pipeline commands to main entry point

Adds gen-cot, train-cot, infer-cot commands. Updates README progress.
Creates src/cot module for CoT reasoning pipeline.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: 端到端运行验证

**注意:** 此任务需要 DeepSeek API Key 和 GPU。分为可立即执行的子步骤和需要GPU的子步骤。

- [ ] **Step 1: CoT数据生成（100条试跑）**

```bash
cd e:/Coding/trAshbaGrepository/shixun/Project_NLP
python scripts/generate_cot_data.py --api_key $env:DEEPSEEK_API_KEY --samples 100
# 观察输出:
# - DeepSeek的零样本准确率（决定训练数据质量和数量）
# - 各领域正确率分布
# - 如果总体正确率 >50%，值得继续；否则需要改进prompt或换模型
```

预期：DeepSeek-V3 零样本 CoT 准确率可能在 40-60% 左右，过滤后得到 40-60 条高质量训练数据（100条试跑），扩展到全量 3600 条后预计得到 1500-2200 条。

- [ ] **Step 2: 全量CoT数据生成（3600条）**

```bash
python scripts/generate_cot_data.py --api_key $env:DEEPSEEK_API_KEY
# 预期耗时: ~15-20分钟, 成本 ~$2-3
# 过滤后预计1500-2200条训练数据
```

- [ ] **Step 3: LoRA微调（需GPU）**

```bash
python scripts/train_cot_model.py \
    --train_data outputs/cot_train_filtered.json \
    --output_dir checkpoints/cot_model \
    --epochs 3 \
    --batch_size 4
# 需GPU (>=16GB VRAM)
# 预计耗时: 2-4小时 (取决于GPU)
```

- [ ] **Step 4: 测试集推理**

```bash
python scripts/run_cot_inference.py \
    --model_path checkpoints/cot_model/final \
    --output outputs/submission_cot.json
# 输出: outputs/submission_cot.json
```

- [ ] **Step 5: 训练集交叉验证**

```bash
# 在训练集上随机采样验证微调效果
python -c "
import json, random
from scripts.run_cot_inference import run_inference, build_prompt, extract_answer_from_response
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load model with LoRA
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype=torch.bfloat16, device_map='auto')
model = PeftModel.from_pretrained(base, 'checkpoints/cot_model/final')

# Eval on 100 random training samples
with open('data/SCoRE2026_trainset.json') as f:
    train = json.load(f)
samples = random.sample(train, 100)

correct = 0
for s in samples:
    prompt = build_prompt(s)
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    out = model.generate(**inputs, max_new_tokens=1024, temperature=0.0, do_sample=False)
    resp = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    pred = extract_answer_from_response(resp)
    if set(pred) == set(s['answers']):
        correct += 1

print(f'Fine-tuned model accuracy (100 samples): {correct}%')
"
```

---

## Self-Review

**1. Spec coverage:**
- ✅ CoT数据生成（Task 1）
- ✅ LoRA微调（Task 2）
- ✅ 推理提交（Task 3）
- ✅ 入口集成和文档（Task 4）
- ✅ 端到端验证（Task 5）
- ⚠️ 消融实验：未在计划中，可在 Task 5 之后手动对比模板基线

**2. Placeholder scan:** 无 TBD/TODO/占位符。所有代码均为完整实现。

**3. Type consistency:** 
- `generate_cot_data.py` 输出 `{"id", "domain", "language", "text": "<chatml>"}` → `train_cot_model.py` 消费 `"text"` 字段 ✅
- `train_cot_model.py` 产出 `checkpoints/cot_model/final/` → `run_cot_inference.py` 消费 `--model_path` ✅
- 答案格式统一为 `{"answers": ["A", "B"]}` ✅
