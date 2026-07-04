"""
约束标注准备脚本

为训练约束解析器准备标注数据：
1. 从训练集中采样
2. 生成Prompt用于LLM辅助标注
3. 验证标注质量（通过符号求解器）
"""
import json
import os
import sys
import random
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def generate_annotation_prompts(samples, output_path):
    """生成用于LLM辅助标注的Prompt"""
    prompts = []

    for sample in samples:
        domain = sample["domain"]
        text = sample["text"]
        question = sample.get("question", "")
        options = sample.get("options", {})
        answers = sample.get("answers", [])

        # 根据领域选择合适的标注prompt
        if domain == "time":
            prompt = _build_time_annotation_prompt(text, question, options, answers)
        elif domain == "space":
            prompt = _build_space_annotation_prompt(text, question, options, answers)
        elif domain == "social":
            prompt = _build_social_annotation_prompt(text, question, options, answers)
        elif domain == "nature":
            prompt = _build_nature_annotation_prompt(text, question, options, answers)
        elif domain == "space+nature":
            prompt = _build_fusion_annotation_prompt(text, question, options, answers, domain)
        else:
            continue

        prompts.append({
            "id": sample["id"],
            "domain": domain,
            "prompt": prompt,
            "expected_answer": answers,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(prompts)} annotation prompts → {output_path}")
    return prompts


def _build_time_annotation_prompt(text, question, options, answers):
    return f"""You are annotating a TEMPORAL reasoning problem. Extract ALL constraints into a structured format.

## Text:
{text}

## Question:
{question}

## Options:
{json.dumps(options, ensure_ascii=False)}

## Correct Answer(s): {answers}

## Your Task:
Extract ALL temporal constraints from the text. Output ONLY valid JSON:

```json
{{
  "domain": "time",
  "time": {{
    "entities": ["list all events mentioned, including those in options"],
    "is_weekly_cycle": true/false,
    "absolute": [
      {{"event": "exact event description", "time_point": "周三 or 星期一 or 1994"}}
    ],
    "relative": [
      {{"event_a": "later event", "event_b": "earlier event", "relation": "after", "offset": number, "unit": "day/year"}},
      {{"event_a": "earlier event", "event_b": "later event", "relation": "before", "offset": number, "unit": "day/year"}}
    ]
  }}
}}
```

Key rules:
- "(1) event N days after reference" → reference is event_b, event is event_a, relation="after", offset=N
- "(1) event N years before reference" → reference is event_b, event is event_a, relation="before", offset=N
- Weekday names (星期一, Wednesday) are absolute time points
- For weekly schedules, set is_weekly_cycle=true
- Include ALL events from the text and option statements"""


def _build_space_annotation_prompt(text, question, options, answers):
    return f"""You are annotating a SPATIAL reasoning problem. Extract ALL constraints into a structured format.

## Text:
{text}

## Question:
{question}

## Options:
{json.dumps(options, ensure_ascii=False)}

## Correct Answer(s): {answers}

## Your Task:
Extract ALL spatial constraints. Output ONLY valid JSON:

```json
{{
  "domain": "space",
  "space": {{
    "structure": "grid_3x2",
    "rows": 3,
    "cols": 2,
    "col_labels": ["东", "西"],
    "entities": ["list all objects"],
    "positions": [
      {{"entity": "name", "row": 0, "col": "东"}}
    ],
    "relations": [
      {{"entity_a": "name", "entity_b": "name", "relation": "above|below|adjacent_left|adjacent_right|same_row|different_col", "gap": 0}}
    ]
  }}
}}
```

Key rules:
- "A在B正上方且隔了N层" → relation="above", gap=N
- "A是B的左邻" → relation="adjacent_left"
- "N层东侧是A" → position with row=N-1, col="东"
- Row 0=top, Row 1=middle, Row 2=bottom
- 东=左=col 0, 西=右=col 1"""


def _build_social_annotation_prompt(text, question, options, answers):
    return f"""You are annotating a SOCIAL RELATIONSHIP reasoning problem.

## Text:
{text}

## Question:
{question}

## Options:
{json.dumps(options, ensure_ascii=False)}

## Correct Answer(s): {answers}

## Your Task:
Extract ALL social/kinship relations. Output ONLY valid JSON:

```json
{{
  "domain": "social",
  "social": {{
    "entities": ["list all person names"],
    "relations": [
      {{"person_a": "name", "person_b": "name", "relation": "父亲/母亲/姐姐/老公/岳母/..."}}
    ]
  }}
}}
```

Key rules:
- "A是B的C" → person_a=A, person_b=B, relation=C
- "A is B's C" → same format
- Include ALL named persons, even those only in options"""


def _build_nature_annotation_prompt(text, question, options, answers):
    return f"""You are annotating a NATURAL ATTRIBUTE reasoning problem.

## Text:
{text}

## Question:
{question}

## Options:
{json.dumps(options, ensure_ascii=False)}

## Correct Answer(s): {answers}

## Your Task:
Extract ALL property and category constraints. Output ONLY valid JSON:

```json
{{
  "domain": "nature",
  "nature": {{
    "entities": ["entity names from text and options"],
    "positions": ["position labels"],
    "entity_properties": {{
      "entity_name": {{"属性名": "属性值"}}
    }},
    "property_constraints": [
      {{"position": "position_label", "property_name": "属性名", "property_value": "属性值"}}
    ],
    "category_constraints": [
      {{"position": "position_label", "category": "类别名"}}
    ]
  }}
}}
```

Key rules:
- Include commonsense knowledge about entity properties
- "N号X的Y是Z" → property_constraint with position="N号X", property_name="Y", property_value="Z"
- "N号X属于C类" → category_constraint"""


def _build_fusion_annotation_prompt(text, question, options, answers, domain):
    sub_domains = domain.split("+")
    return f"""You are annotating a FUSION commonsense reasoning problem combining {', '.join(sub_domains)}.

## Text:
{text}

## Question:
{question}

## Options:
{json.dumps(options, ensure_ascii=False)}

## Correct Answer(s): {answers}

## Your Task:
Extract constraints for BOTH domains. Output ONLY valid JSON:

```json
{{
  "domain": "{domain}",
  "{sub_domains[0]}": {{ /* constraints for {sub_domains[0]} domain */ }},
  "{sub_domains[1]}": {{ /* constraints for {sub_domains[1]} domain */ }}
}}
```

IMPORTANT: The text first describes {sub_domains[0]} relationships, then uses those to describe {sub_domains[1]} constraints. Extract both carefully."""


def validate_annotations(annotations_path, solver_check=True):
    """通过符号求解器验证约束标注的质量"""
    print(f"\nValidating annotations: {annotations_path}")

    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    valid_count = 0
    invalid_ids = []

    for ann in annotations:
        # 基本格式检查
        if "output" not in ann or "constraints" not in ann["output"]:
            invalid_ids.append(ann.get("id", "unknown"))
            continue

        constraints = ann["output"]["constraints"]
        if not isinstance(constraints, dict) or "domain" not in constraints:
            invalid_ids.append(ann["id"])
            continue

        valid_count += 1

    print(f"  Valid: {valid_count}/{len(annotations)}")
    if invalid_ids:
        print(f"  Invalid IDs: {invalid_ids[:10]}...")
    return valid_count, invalid_ids


def main():
    parser = argparse.ArgumentParser(description="Prepare constraint annotation data")
    parser.add_argument("--samples_per_domain", type=int, default=50,
                       help="Number of samples to annotate per domain")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for annotation prompts")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = args.output or os.path.join(OUTPUT_DIR, "annotation_prompts.json")

    # 加载训练数据
    with open(os.path.join(DATA_DIR, "SCoRE2026_trainset.json"), "r", encoding="utf-8") as f:
        train = json.load(f)

    # 按领域采样
    samples = []
    for domain in ["time", "space", "social", "nature", "space+nature"]:
        domain_samples = [s for s in train if s["domain"] == domain]
        sampled = random.sample(domain_samples,
                               min(args.samples_per_domain, len(domain_samples)))
        samples.extend(sampled)
        print(f"  {domain}: sampled {len(sampled)}/{len(domain_samples)}")

    print(f"\nTotal samples to annotate: {len(samples)}")

    # 生成标注prompt
    prompts = generate_annotation_prompts(samples, output_path)

    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print(f"1. Review prompts in {output_path}")
    print("2. Use a strong LLM (GPT-4/Claude/DeepSeek) to generate constraint JSONs")
    print("3. Save results as outputs/constraint_annotations.json")
    print("4. Run validation: python scripts/prepare_constraint_labels.py --validate")
    print("5. Train: python scripts/train_constraint_parser.py")


if __name__ == "__main__":
    random.seed(42)
    main()
