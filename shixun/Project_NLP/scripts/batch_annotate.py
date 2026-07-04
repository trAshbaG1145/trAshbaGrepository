"""
批量约束标注生成脚本

使用强LLM（DeepSeek/OpenAI/Claude）批量将自然语言场景转化为结构化约束JSON，
并通过符号求解器反向验证标注质量。

用法:
    # 使用 DeepSeek API（推荐，性价比高）
    python scripts/batch_annotate.py --api deepseek --api_key sk-xxx

    # 使用 OpenAI API
    python scripts/batch_annotate.py --api openai --api_key sk-xxx

    # 使用 Anthropic Claude API
    python scripts/batch_annotate.py --api anthropic --api_key sk-xxx

    # 高级选项
    python scripts/batch_annotate.py --api deepseek --api_key sk-xxx \
        --model deepseek-chat \
        --concurrency 5 \
        --resume \
        --validate_only \
        --samples 50

流程:
    Phase 1: LLM API批量调用 → outputs/constraint_annotations_raw.json
    Phase 2: 符号求解器验证 → 统计报告 + 筛选高质量标注
    Phase 3: 输出 outputs/constraint_annotations.json（训练用格式）
"""

import json
import os
import sys
import re
import time
import argparse
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
CACHE_DIR = os.path.join(OUTPUT_DIR, ".annotation_cache")

# ================================================================
# API 客户端封装
# ================================================================


class LLMClient:
    """统一的LLM API客户端"""

    def __init__(self, api_type: str, api_key: str, model: str = None,
                 base_url: str = None, max_tokens: int = 2048,
                 temperature: float = 0.0):
        self.api_type = api_type
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature

        # 默认模型和base_url
        defaults = {
            "deepseek": {
                "model": "deepseek-chat",
                "base_url": "https://api.deepseek.com",
            },
            "openai": {
                "model": "gpt-4o",
                "base_url": "https://api.openai.com/v1",
            },
            "anthropic": {
                "model": "claude-sonnet-4-20250514",
                "base_url": None,
            },
        }

        cfg = defaults.get(api_type, {})
        self.model = model or cfg.get("model", "deepseek-chat")
        self.base_url = base_url or cfg.get("base_url")

        self._init_client()

    def _init_client(self):
        """初始化对应的API客户端"""
        if self.api_type in ("deepseek", "openai"):
            try:
                from openai import OpenAI
            except ImportError:
                print("请安装 openai 包: pip install openai")
                sys.exit(1)
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        elif self.api_type == "anthropic":
            try:
                import anthropic
            except ImportError:
                print("请安装 anthropic 包: pip install anthropic")
                sys.exit(1)
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"不支持的API类型: {self.api_type}")

    def chat(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """发送聊天请求，返回响应文本"""
        if self.api_type in ("deepseek", "openai"):
            return self._chat_openai(system_prompt, user_prompt)
        elif self.api_type == "anthropic":
            return self._chat_anthropic(system_prompt, user_prompt)

    def _chat_openai(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def _chat_anthropic(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text


# ================================================================
# JSON提取与解析
# ================================================================


def extract_json_from_response(response: str) -> Optional[Dict]:
    """从LLM响应中提取JSON对象"""
    if not response:
        return None

    # 策略1: ```json ... ``` 代码块
    match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 策略2: 整个响应是JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 策略3: 匹配最外层 {...}
    brace_match = re.search(r'\{.*\}', response, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ================================================================
# 标注验证器
# ================================================================


class AnnotationValidator:
    """用符号求解器验证约束标注是否正确"""

    def __init__(self):
        from src.pipeline.score_pipeline import SCOREPipeline
        from src.parser.constraint_parser import TemplateConstraintParser
        # 这里不需要TemplateParser，我们直接使用StructuredConstraints
        self.pipeline = SCOREPipeline(parser=None)

    def validate(self, constraints_dict: Dict, sample_info: Dict) -> Dict:
        """
        验证约束标注质量

        Args:
            constraints_dict: LLM生成的约束JSON字典
            sample_info: 包含 domain, question, options, expected_answer 的字典

        Returns:
            {"valid": bool, "correct": bool, "predicted": [], "expected": [],
             "error": str, "solution": {}}
        """
        result = {
            "valid": False,
            "correct": False,
            "predicted": [],
            "expected": sample_info.get("expected_answer", []),
            "error": None,
            "solution": {},
        }

        try:
            # Step 1: 转换为 StructuredConstraints
            from src.constraint_schema import StructuredConstraints
            from src.parser.constraint_parser import LLMConstraintParser

            # 创建一个临时的dummy解析器用于_dict_to_constraints
            parser = LLMConstraintParser.__new__(LLMConstraintParser)
            structured = parser._dict_to_constraints(
                constraints_dict,
                domain=sample_info["domain"],
                text="",
                language=sample_info.get("language", "cn"),
            )

            # Step 2: 求解
            domain = sample_info["domain"]
            question = sample_info.get("question", "")
            options = sample_info.get("options", {})

            if "+" in domain:
                if self.pipeline.fusion_solver:
                    fusion_result = self.pipeline.fusion_solver.solve(structured)
                    solution = {}
                    for sub_domain, sub_result in fusion_result.get("results", {}).items():
                        solution.update(sub_result)
                else:
                    result["error"] = "Fusion solver not available"
                    return result
            else:
                base_domain = domain.split("+")[0]
                solver = self.pipeline.solvers.get(base_domain)
                if not solver:
                    result["error"] = f"No solver for {base_domain}"
                    return result
                solution = self.pipeline._solve_single_domain(
                    solver, structured, base_domain
                )

            result["solution"] = solution

            # Step 3: 验证答案
            predicted = self.pipeline.verifier.verify(
                question, options, solution, domain
            )
            result["predicted"] = predicted
            result["valid"] = True
            result["correct"] = set(predicted) == set(result["expected"])

        except Exception as e:
            result["error"] = str(e)

        return result


# ================================================================
# 缓存管理
# ================================================================


def _cache_key(prompt: str, model: str) -> str:
    """生成缓存键"""
    content = f"{model}|{prompt}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def load_cache() -> Dict:
    """加载已有缓存"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, "responses.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict):
    """保存缓存"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, "responses.json")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ================================================================
# 主标注流程
# ================================================================


def load_train_data() -> Dict[str, Dict]:
    """加载训练集，建立 id -> sample 映射"""
    train_path = os.path.join(DATA_DIR, "SCoRE2026_trainset.json")
    with open(train_path, "r", encoding="utf-8") as f:
        train = json.load(f)
    return {s["id"]: s for s in train}


def build_sample_info(prompt_entry: Dict, train_map: Dict) -> Dict:
    """从prompt条目构建完整的样本信息"""
    sample = train_map.get(prompt_entry["id"], {})
    return {
        "id": prompt_entry["id"],
        "domain": prompt_entry["domain"],
        "language": sample.get("language", "cn"),
        "text": sample.get("text", ""),
        "question": sample.get("question", ""),
        "options": sample.get("options", {}),
        "expected_answer": prompt_entry.get("expected_answer", []),
    }


def process_prompts(
    client: LLMClient,
    prompts: List[Dict],
    train_map: Dict[str, Dict],
    concurrency: int = 3,
    resume: bool = True,
    dry_run: bool = False,
) -> List[Dict]:
    """
    批量处理标注prompt

    Returns:
        [{"id", "domain", "sample_info": {...}, "raw_response": str,
          "constraints_json": {...}, "validation": {...}}]
    """
    cache = load_cache() if resume else {}
    results = []
    lock = __import__('threading').Lock()

    def process_one(entry: Dict) -> Optional[Dict]:
        prompt_text = entry["prompt"]
        ck = _cache_key(prompt_text, client.model)

        # 检查缓存
        with lock:
            if resume and ck in cache:
                return cache[ck]

        if dry_run:
            print(f"  [DRY RUN] Would call API for {entry['id']} ({entry['domain']})")
            return None

        # 拆分system/user
        system_prompt = (
            "You are an expert at extracting structured logical constraints from "
            "commonsense reasoning scenarios. Your task is to read the scenario text "
            "and convert ALL stated facts into a precise JSON constraint representation.\n\n"
            "CRITICAL RULES:\n"
            "1. Copy entity/event names EXACTLY as they appear in the text - "
            "do NOT paraphrase, shorten, or translate them.\n"
            "2. Only include events/facts STATED in the text - do NOT add option text as entities.\n"
            "3. Duration/deadline statements (e.g. 'he studied for 3 years') are NOT events.\n"
            "4. For relative constraints: event_a is the one whose time is being described, "
            "event_b is the reference point. 'A 2 days after B' means event_a=A, event_b=B, "
            "relation=after, offset=2.\n"
            "5. For CN text: '在A之后N天，B' means B is N days AFTER A. "
            "event_a=B (the target), event_b=A (the reference), relation=after.\n"
            "6. For EN text: 'A N days after B' means A is N days AFTER B. "
            "event_a=A, event_b=B, relation=after.\n"
            "7. Every stated constraint must be extracted - do NOT skip any numbered item.\n"
            "8. Output ONLY the JSON object - no markdown fences, no explanation, no comments.\n"
            "9. For weekly cycle problems (days of week), set is_weekly_cycle=true and use "
            "0=Monday through 6=Sunday for weekday values.\n"
            "10. Include events from the question if they refer to entities in the text.\n\n"
            "EXAMPLE - Temporal (EN):\n"
            "Text: (1)On Wednesday, Jack plays badminton; (2)Jack practices guitar 2 days after he plays badminton.\n"
            'Output: {"domain":"time","time":{"entities":["Jack plays badminton","Jack practices guitar"],'
            '"is_weekly_cycle":true,"absolute":[{"event":"Jack plays badminton","time_point":"Wednesday"}],'
            '"relative":[{"event_a":"Jack practices guitar","event_b":"Jack plays badminton","relation":"after","offset":2,"unit":"day"}]}}\n\n'
            "EXAMPLE - Temporal (CN):\n"
            "Text: (1)星期三，他打羽毛球; (2)在他打羽毛球之后2天，他练习吉他。\n"
            'Output: {"domain":"time","time":{"entities":["他打羽毛球","他练习吉他"],'
            '"is_weekly_cycle":true,"absolute":[{"event":"他打羽毛球","time_point":"星期三"}],'
            '"relative":[{"event_a":"他练习吉他","event_b":"他打羽毛球","relation":"after","offset":2,"unit":"day"}]}}\n\n'
            "REMEMBER: Output ONLY the JSON. Entity names MUST match the text exactly."
        )

        # 调用API（带重试）
        max_retries = 3
        raw_response = None
        for attempt in range(max_retries):
            try:
                raw_response = client.chat(system_prompt, prompt_text)
                break
            except Exception as e:
                print(f"  API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        if raw_response is None:
            return None

        # 提取JSON
        constraints_json = extract_json_from_response(raw_response)

        sample_info = build_sample_info(entry, train_map)
        result = {
            "id": entry["id"],
            "domain": entry["domain"],
            "sample_info": sample_info,
            "raw_response": raw_response,
            "constraints_json": constraints_json,
        }

        # 存入缓存
        with lock:
            cache[ck] = result
            if len(cache) % 20 == 0:
                save_cache(cache)

        return result

    # 并发处理
    completed = 0
    total = len(prompts)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_entry = {
            executor.submit(process_one, entry): entry for entry in prompts
        }

        for future in as_completed(future_to_entry):
            entry = future_to_entry[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                completed += 1
                status = "✓" if (result and result.get("constraints_json")) else "✗"
                print(f"  [{completed}/{total}] {status} {entry['id']} ({entry['domain']})")
            except Exception as e:
                completed += 1
                print(f"  [{completed}/{total}] ✗ {entry['id']} ERROR: {e}")

    # 最终保存缓存
    save_cache(cache)
    return results


def postprocess_constraints(results: List[Dict], train_map: Dict[str, Dict]) -> List[Dict]:
    """后处理：清理约束JSON中的无效实体"""
    import re
    for r in results:
        cj = r.get("constraints_json")
        if not cj:
            continue

        if "time" in cj:
            t = cj["time"]
            # Remove non-event entities: standalone years, duration descriptions
            t["entities"] = [
                e for e in t.get("entities", [])
                if not re.match(r'^\d{4}年?$', e)
                and not re.search(r'一共?\d+年', e)
                and not re.search(r'\d+\s*years?\s*$', e)
            ]
            t["absolute"] = [
                a for a in t.get("absolute", [])
                if a.get("event") and not re.match(r'^\d{4}年?$', a["event"])
            ]

        if "space" in cj:
            sp = cj["space"]
            sp["entities"] = [
                e for e in sp.get("entities", []) if len(e) < 80
            ]

        if "social" in cj:
            soc = cj["social"]
            soc["entities"] = [
                e for e in soc.get("entities", [])
                if re.match(r'^[一-鿿]{2,4}$', e)
                or re.match(r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)?$', e)
            ]

    return results


def validate_results(results: List[Dict]) -> List[Dict]:
    """用符号求解器验证标注"""
    validator = AnnotationValidator()
    validated = []

    for r in results:
        if not r.get("constraints_json"):
            r["validation"] = {"valid": False, "correct": False,
                              "predicted": [], "expected": [],
                              "error": "No constraints extracted", "solution": {}}
            validated.append(r)
            continue

        validation = validator.validate(
            r["constraints_json"],
            r.get("sample_info", {}),
        )
        r["validation"] = validation
        validated.append(r)

    return validated


def print_stats(results: List[Dict]):
    """打印标注统计"""
    total = len(results)
    extracted = sum(1 for r in results if r.get("constraints_json"))
    valid = sum(1 for r in results if r.get("validation", {}).get("valid"))
    correct = sum(1 for r in results if r.get("validation", {}).get("correct"))

    print("\n" + "=" * 60)
    print("标注统计")
    print("=" * 60)
    print(f"  总计:        {total}")
    print(f"  JSON提取成功: {extracted} ({extracted/max(total,1)*100:.1f}%)")
    print(f"  求解成功:    {valid} ({valid/max(total,1)*100:.1f}%)")
    print(f"  答案正确:    {correct} ({correct/max(total,1)*100:.1f}%)")
    print(f"  高质量率:    {correct}/{extracted} (提取成功中的 {correct/max(extracted,1)*100:.1f}%)")

    # 分领域
    by_domain = defaultdict(lambda: {"total": 0, "extracted": 0, "valid": 0, "correct": 0})
    for r in results:
        d = r["domain"]
        by_domain[d]["total"] += 1
        if r.get("constraints_json"):
            by_domain[d]["extracted"] += 1
        if r.get("validation", {}).get("valid"):
            by_domain[d]["valid"] += 1
        if r.get("validation", {}).get("correct"):
            by_domain[d]["correct"] += 1

    print("\n--- 分领域统计 ---")
    print(f"  {'领域':15s} {'总数':>5s} {'提取':>5s} {'求解成功':>7s} {'答案正确':>7s} {'正确率':>7s}")
    for domain in sorted(by_domain):
        ds = by_domain[domain]
        acc = ds["correct"] / max(ds["extracted"], 1) * 100
        print(f"  {domain:15s} {ds['total']:5d} {ds['extracted']:5d} "
              f"{ds['valid']:7d} {ds['correct']:7d} {acc:6.1f}%")

    # 错误分析
    errors = [r for r in results
              if r.get("validation", {}).get("valid") and not r.get("validation", {}).get("correct")]
    if errors:
        print(f"\n--- 求解正确但答案错误 ({len(errors)}条) ---")
        print("  (说明约束JSON不完整/有误)")
        for e in errors[:5]:
            v = e.get("validation", {})
            print(f"  {e['id']} [{e['domain']}] pred={v.get('predicted')} "
                  f"expected={v.get('expected')}")

    # 未提取JSON的
    no_json = [r for r in results if not r.get("constraints_json")]
    if no_json:
        print(f"\n--- JSON提取失败 ({len(no_json)}条) ---")
        for r in no_json[:3]:
            resp = r.get("raw_response", "")
            print(f"  {r['id']}: {resp[:200]}...")


def export_training_data(results: List[Dict], output_path: str,
                         quality_filter: bool = True):
    """
    导出训练数据，格式匹配 train_constraint_parser.py 的预期

    Args:
        results: 标注结果
        output_path: 输出路径
        quality_filter: 是否只导出通过验证的高质量标注
    """
    train_data = []

    for r in results:
        constraints = r.get("constraints_json")
        if not constraints:
            continue

        validation = r.get("validation", {})
        if quality_filter and not validation.get("correct", False):
            continue

        info = r.get("sample_info", {})
        train_data.append({
            "id": r["id"],
            "domain": r["domain"],
            "language": info.get("language", "cn"),
            "input": {
                "text": info.get("text", ""),
                "question": info.get("question", ""),
                "domain": r["domain"],
            },
            "output": {
                "constraints": constraints,
            },
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    print(f"\n导出训练数据: {len(train_data)} 条 → {output_path}")
    return train_data


def export_raw_results(results: List[Dict], output_path: str):
    """导出完整的原始结果（含验证信息，用于调试）"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 序列化时不包含不可序列化的对象
    serializable = []
    for r in results:
        sr = {
            "id": r["id"],
            "domain": r["domain"],
            "sample_info": r.get("sample_info", {}),
            "constraints_json": r.get("constraints_json"),
            "validation": {
                "valid": r.get("validation", {}).get("valid"),
                "correct": r.get("validation", {}).get("correct"),
                "predicted": r.get("validation", {}).get("predicted", []),
                "expected": r.get("validation", {}).get("expected", []),
                "error": r.get("validation", {}).get("error"),
            },
        }
        serializable.append(sr)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"原始结果已保存: {len(serializable)} 条 → {output_path}")


# ================================================================
# CLI
# ================================================================


def main():
    parser = argparse.ArgumentParser(
        description="批量约束标注生成 - 使用LLM生成训练数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/batch_annotate.py --api deepseek --api_key sk-xxx
  python scripts/batch_annotate.py --api openai --api_key sk-xxx --model gpt-4o
  python scripts/batch_annotate.py --api anthropic --api_key sk-xxx
  python scripts/batch_annotate.py --api deepseek --api_key sk-xxx --validate_only
        """,
    )
    parser.add_argument("--api", type=str, default="deepseek",
                       choices=["deepseek", "openai", "anthropic"],
                       help="LLM API 类型 (默认: deepseek)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="API Key（也可通过环境变量设置，见下方）")
    parser.add_argument("--model", type=str, default=None,
                       help="模型名称（默认按API类型自动选择）")
    parser.add_argument("--base_url", type=str, default=None,
                       help="自定义API base URL")
    parser.add_argument("--prompts_file", type=str,
                       default=None,
                       help="标注prompt文件路径（默认: outputs/annotation_prompts.json）")
    parser.add_argument("--concurrency", type=int, default=3,
                       help="并发请求数 (默认: 3)")
    parser.add_argument("--samples", type=int, default=0,
                       help="只处理前N条 (0=全部)")
    parser.add_argument("--domain", type=str, default=None,
                       help="只处理特定领域 (time/space/social/nature/space+nature)")
    parser.add_argument("--no_resume", action="store_true",
                       help="不使用缓存，重新请求全部")
    parser.add_argument("--dry_run", action="store_true",
                       help="只显示将要处理的条目，不实际调用API")
    parser.add_argument("--validate_only", action="store_true",
                       help="仅验证已有的原始结果，不调用API")
    parser.add_argument("--quality_filter", action="store_true", default=True,
                       help="只导出通过验证的高质量标注 (默认开启)")
    parser.add_argument("--no_quality_filter", action="store_true",
                       help="导出全部标注（包括未通过验证的）")
    parser.add_argument("--output", type=str, default=None,
                       help="训练数据输出路径（默认: outputs/constraint_annotations.json）")

    args = parser.parse_args()

    # API Key: 优先命令行，其次环境变量
    env_key_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    api_key = args.api_key or os.environ.get(env_key_map.get(args.api, ""))

    if not api_key and not args.validate_only and not args.dry_run:
        print(f"错误: 需要提供 --api_key 或设置环境变量 {env_key_map.get(args.api, 'API_KEY')}")
        sys.exit(1)

    # 加载prompt文件
    prompts_file = args.prompts_file or os.path.join(OUTPUT_DIR, "annotation_prompts.json")
    if not os.path.exists(prompts_file):
        print(f"错误: Prompt文件不存在: {prompts_file}")
        print("请先运行: python scripts/prepare_constraint_labels.py")
        sys.exit(1)

    with open(prompts_file, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)
    print(f"加载 {len(all_prompts)} 条标注prompt")

    # 过滤
    prompts = all_prompts
    if args.domain:
        prompts = [p for p in prompts if p["domain"] == args.domain]
        print(f"  过滤领域 '{args.domain}': {len(prompts)} 条")
    if args.samples > 0:
        prompts = prompts[:args.samples]
        print(f"  采样前 {args.samples} 条")

    if not prompts:
        print("无待处理条目")
        return

    # 加载训练数据
    train_map = load_train_data()
    print(f"加载训练集映射: {len(train_map)} 条")

    output_path = args.output or os.path.join(OUTPUT_DIR, "constraint_annotations.json")
    raw_path = os.path.join(OUTPUT_DIR, "constraint_annotations_raw.json")

    # Validate only 模式
    if args.validate_only:
        print("\n=== 仅验证模式 ===")
        if not os.path.exists(raw_path):
            print(f"错误: 原始结果文件不存在: {raw_path}")
            print("请先运行标注生成")
            sys.exit(1)
        with open(raw_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        results = validate_results(results)
        print_stats(results)
        quality_filter = not args.no_quality_filter
        export_training_data(results, output_path, quality_filter=quality_filter)
        return

    # Dry run 模式
    if args.dry_run:
        print(f"\n=== Dry Run: {len(prompts)} 条待处理 ===")
        for p in prompts[:10]:
            print(f"  {p['id']} ({p['domain']})")
        if len(prompts) > 10:
            print(f"  ... 还有 {len(prompts) - 10} 条")
        return

    # === 主流程 ===
    print(f"\n{'=' * 60}")
    print(f"Phase 1: LLM API 批量调用")
    print(f"{'=' * 60}")
    print(f"  API: {args.api} | Model: {args.model or '默认'}")
    print(f"  并发: {args.concurrency} | 条目: {len(prompts)}")
    print(f"  断点续传: {'否' if args.no_resume else '是'}")
    print()

    client = LLMClient(
        api_type=args.api,
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
    )

    start_time = time.time()
    results = process_prompts(
        client=client,
        prompts=prompts,
        train_map=train_map,
        concurrency=args.concurrency,
        resume=not args.no_resume,
        dry_run=False,
    )
    elapsed = time.time() - start_time
    print(f"\nPhase 1 完成: {len(results)} 条 | 耗时 {elapsed:.0f}s "
          f"({elapsed/max(len(results),1):.1f}s/条)")

    # 保存原始结果
    export_raw_results(results, raw_path)

    # === Phase 1.5: 后处理 ===
    print(f"\nPhase 1.5: 后处理（清洗无效实体）")
    results = postprocess_constraints(results, train_map)
    print(f"  后处理完成")

    # === Phase 2: 验证 ===
    print(f"\n{'=' * 60}")
    print(f"Phase 2: 符号求解器验证")
    print(f"{'=' * 60}")

    results = validate_results(results)
    print_stats(results)

    # === Phase 3: 导出训练数据 ===
    print(f"\n{'=' * 60}")
    print(f"Phase 3: 导出训练数据")
    print(f"{'=' * 60}")

    quality_filter = not args.no_quality_filter
    train_data = export_training_data(results, output_path, quality_filter=quality_filter)

    # 后续步骤提示
    print(f"\n{'=' * 60}")
    print("下一步")
    print(f"{'=' * 60}")
    if train_data:
        print(f"1. 训练约束解析器:")
        print(f"   python scripts/train_constraint_parser.py \\")
        print(f"     --train_data {output_path} \\")
        print(f"     --output_dir checkpoints/constraint_parser")
        print(f"2. (可选) 扩展标注到全量3600条:")
        print(f"   重新运行 prepare_constraint_labels.py 采样更多数据")
        print(f"   将已验证的高质量标注作为few-shot示例")
    else:
        print("⚠️  没有通过验证的高质量标注，建议:")
        print("  1. 检查原始结果 outputs/constraint_annotations_raw.json")
        print("  2. 尝试更换模型 (--model gpt-4o 或 --api anthropic)")
        print("  3. 改进prompt模板 (编辑 scripts/prepare_constraint_labels.py)")


if __name__ == "__main__":
    main()
