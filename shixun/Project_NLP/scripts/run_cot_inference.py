"""
CoT模型推理脚本

加载微调后的Qwen2.5-7B模型，在测试集上执行CoT推理并生成提交文件。

用法:
    python scripts/run_cot_inference.py \
        --model_path checkpoints/cot_model/final \
        --output outputs/submission_cot.json

    # 零样本模式（无LoRA权重时使用基模型直接推理）
    python scripts/run_cot_inference.py --zeroshot
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

    # 策略2: "answers": ["A", "B"] 片段
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

    # 策略4: 最后几行中的选项字母
    lines = response.strip().split('\n')
    for line in reversed(lines[-5:]):
        found = re.findall(r'\b([A-D])\b', line)
        if found and len(found) <= 4:
            return found

    return []


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


def build_prompt(sample: Dict) -> str:
    """构建推理prompt（ChatML格式，与训练时一致）"""
    options_text = "\n".join(
        f"{l}: {sample['options'][l]}" for l in sorted(sample['options'].keys())
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


def run_inference(model, tokenizer, samples: List[Dict],
                  batch_size: int = 1, max_new_tokens: int = 1024) -> List[Dict]:
    """批量推理"""
    import torch
    results = []

    for i, sample in enumerate(samples):
        prompt = build_prompt(sample)
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
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

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(samples)}] {sample['id']}: {predicted}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="CoT Model Inference for SCoRE2026")
    parser.add_argument("--model_path", type=str,
                       default="checkpoints/cot_model/final",
                       help="Path to LoRA adapter")
    parser.add_argument("--base_model", type=str,
                       default="Qwen/Qwen2.5-7B-Instruct",
                       help="Base model name")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for submission JSON")
    parser.add_argument("--samples", type=int, default=0,
                       help="Max test samples (0=all 1000)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                       help="Max tokens to generate")
    parser.add_argument("--zeroshot", action="store_true",
                       help="Use base model without LoRA (zero-shot baseline)")
    parser.add_argument("--eval_train", action="store_true",
                       help="Evaluate on training set instead of test set")
    args = parser.parse_args()

    print(f"Loading model: {args.base_model}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install: pip install transformers peft accelerate")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.zeroshot:
        print("Zero-shot mode: using base model without LoRA")
        model = base_model
    elif os.path.exists(args.model_path):
        print(f"Loading LoRA adapter: {args.model_path}")
        model = PeftModel.from_pretrained(base_model, args.model_path)
        print("LoRA adapter loaded")
    else:
        print(f"WARNING: LoRA adapter not found at {args.model_path}")
        print("Falling back to zero-shot mode")
        model = base_model

    model.eval()

    # 加载数据
    if args.eval_train:
        data_path = os.path.join(DATA_DIR, "SCoRE2026_trainset.json")
        print("Evaluating on TRAINING set")
    else:
        data_path = os.path.join(DATA_DIR, "SCoRE2026_testset.json")
        print("Evaluating on TEST set")

    with open(data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    if args.samples > 0:
        eval_data = eval_data[:args.samples]

    print(f"Running inference on {len(eval_data)} samples...")

    # 推理
    results = run_inference(model, tokenizer, eval_data,
                           max_new_tokens=args.max_new_tokens)

    # 保存
    output_path = args.output or os.path.join(OUTPUT_DIR, "submission_cot.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    non_empty = sum(1 for r in results if r["answers"])
    print(f"\nSubmission saved to {output_path}")
    print(f"Total: {len(results)}, Non-empty: {non_empty} "
          f"({non_empty/max(len(results),1)*100:.1f}%)")

    # 答案分布
    answer_counts = Counter()
    for r in results:
        for a in r["answers"]:
            answer_counts[a] += 1
    print(f"Answer distribution: {dict(sorted(answer_counts.items()))}")

    # 如果评估训练集，计算准确率
    if args.eval_train:
        correct = 0
        for r, s in zip(results, eval_data):
            if set(r["answers"]) == set(s.get("answers", [])):
                correct += 1
        acc = correct / max(len(eval_data), 1) * 100
        print(f"Training set accuracy: {correct}/{len(eval_data)} ({acc:.1f}%)")


if __name__ == "__main__":
    main()
