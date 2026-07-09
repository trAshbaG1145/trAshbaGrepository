"""
CoT 推理数据生成脚本

用 DeepSeek API 在训练集上生成 Chain-of-Thought 推理链，
按正确答案过滤后作为 Qwen2.5-7B 的微调数据。

用法:
    python scripts/generate_cot_data.py --api_key sk-xxx --samples 100
    python scripts/generate_cot_data.py --api_key sk-xxx  # 全量3600条
    python scripts/generate_cot_data.py --samples 10  # 用环境变量 DEEPSEEK_API_KEY
"""
from dotenv import load_dotenv
load_dotenv()

import json
import os
import sys
import re
import time
import argparse
import hashlib
from typing import Dict, List, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
CACHE_DIR = os.path.join(OUTPUT_DIR, ".cot_cache")

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
    # 策略1: JSON格式 {"answers": ["A", "B"]}
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


def extract_reasoning_from_response(response: str) -> str:
    """从响应中提取推理部分"""
    match = re.search(r'##\s*Reasoning\s*\n(.*?)(?=##\s*Answer|\Z)', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    answer_match = re.search(r'##\s*Answer', response, re.IGNORECASE)
    if answer_match:
        return response[:answer_match.start()].strip()
    return response.strip()


# ============================================================
# API 调用
# ============================================================

class CoTDataGenerator:
    """CoT训练数据生成器（支持断点续传）"""

    def __init__(self, api_key: str, model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com",
                 resume: bool = True):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.resume = resume
        self.cache: Dict[str, Dict] = {}
        self._lock = __import__('threading').RLock()  # 可重入锁，避免 _save_cache 死锁
        if resume:
            self._load_cache()

    def _cache_key(self, sample_id: str) -> str:
        content = f"{self.model}|{sample_id}"
        return hashlib.md5(content.encode()).hexdigest()

    def _cache_path(self) -> str:
        os.makedirs(CACHE_DIR, exist_ok=True)
        return os.path.join(CACHE_DIR, "responses.json")

    def _load_cache(self):
        path = self._cache_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"  Loaded {len(self.cache)} cached responses (resume enabled)")
            except (json.JSONDecodeError, IOError):
                self.cache = {}

    def _save_cache(self):
        path = self._cache_path()
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False)

    def generate_one(self, sample: Dict) -> Optional[Dict]:
        """为单条样本生成CoT推理（优先使用缓存）"""
        ck = self._cache_key(sample["id"])

        # 检查缓存
        if self.resume and ck in self.cache:
            cached = self.cache[ck]
            if cached and cached.get("id") == sample["id"]:
                return cached

        # 调用API
        options_text = format_options(sample["options"])
        user_prompt = COT_USER_TEMPLATE.format(
            text=sample["text"],
            question=sample["question"],
            options_text=options_text,
        )

        max_retries = 3
        full_response = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": COT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=2048,
                    temperature=0.0,
                    timeout=60.0,  # 60s 超时
                )
                full_response = response.choices[0].message.content
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise e

        reasoning = extract_reasoning_from_response(full_response)
        predicted = extract_answer_from_response(full_response)
        ground_truth = sample.get("answers", [])

        result = {
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

        # 存入缓存
        if self.resume:
            with self._lock:
                self.cache[ck] = result
                # 每20条保存一次缓存
                if len(self.cache) % 20 == 0:
                    self._save_cache()

        return result

    def generate_batch(self, samples: List[Dict], concurrency: int = 5) -> List[Dict]:
        """批量生成CoT数据"""
        results = []
        completed = 0
        total = len(samples)

        # 从缓存中恢复已完成的结果
        if self.resume:
            cached_results = []
            pending = []
            for s in samples:
                ck = self._cache_key(s["id"])
                if ck in self.cache and self.cache[ck].get("id") == s["id"]:
                    cached_results.append(self.cache[ck])
                else:
                    pending.append(s)
            if cached_results:
                results.extend(cached_results)
                completed = len(cached_results)
                print(f"  Resumed {completed} cached results, {len(pending)} remaining")
            samples = pending

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
                        status = "OK" if result["is_correct"] else "WRONG"
                    else:
                        status = "ERROR"
                        continue
                except Exception as e:
                    print(f"  [{completed}/{total}] ERROR {sample['id']}: {e}")
                    continue

                print(f"  [{completed}/{total}] {status} {sample['id']} "
                      f"[{sample['domain']}] pred={result.get('predicted_answer', [])} "
                      f"actual={sample.get('answers', [])}")

        # 最终保存缓存
        if self.resume:
            self._save_cache()
            print(f"  Cache saved: {len(self.cache)} entries")

        return results


# ============================================================
# 数据导出
# ============================================================

def export_cot_train_data(results: List[Dict], output_path: str,
                          quality_filter: bool = True) -> List[Dict]:
    """导出CoT训练数据为Qwen2.5的ChatML格式"""
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


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate CoT reasoning data for SCoRE2026")
    parser.add_argument("--api_key", type=str, default=None,
                       help="DeepSeek API Key (or set DEEPSEEK_API_KEY env var)")
    parser.add_argument("--model", type=str, default="deepseek-chat",
                       help="Model name (default: deepseek-chat)")
    parser.add_argument("--samples", type=int, default=0,
                       help="Max samples to process (0=all 3600)")
    parser.add_argument("--concurrency", type=int, default=5,
                       help="Concurrent API calls (default: 5)")
    parser.add_argument("--no_resume", action="store_true",
                       help="Don't use cache, re-fetch all samples")
    parser.add_argument("--domain", type=str, default=None,
                       help="Only process specific domain")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for filtered training data")
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

    resume_str = "" if args.no_resume else " (resume enabled)"
    print(f"Processing {len(train_data)} samples with model '{args.model}'{resume_str}...")

    # 生成
    generator = CoTDataGenerator(api_key=api_key, model=args.model,
                                  resume=not args.no_resume)
    start = time.time()
    results = generator.generate_batch(train_data, concurrency=args.concurrency)
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/max(len(results),1):.1f}s/sample)")

    # 统计
    print_stats(results)

    # 导出原始结果
    raw_path = args.raw_output or os.path.join(OUTPUT_DIR, "cot_train_raw.json")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nRaw results saved to {raw_path}")

    # 导出过滤后的训练数据
    output_path = args.output or os.path.join(OUTPUT_DIR, "cot_train_filtered.json")
    train_export = export_cot_train_data(results, output_path, quality_filter=True)
    print(f"Training data ({len(train_export)} samples) saved to {output_path}")

    # 后续步骤
    if train_export:
        print(f"\nNext: LoRA fine-tune with {len(train_export)} training samples")
        print(f"  python scripts/train_cot_model.py \\")
        print(f"    --train_data {output_path} \\")
        print(f"    --output_dir checkpoints/cot_model")


if __name__ == "__main__":
    main()
