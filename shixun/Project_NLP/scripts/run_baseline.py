"""
基线测试脚本
测试纯LLM（零样本/Few-shot CoT）和Neuro-Symbolic方案的性能
"""
import json
import sys
import os
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.score_pipeline import SCOREPipeline
from src.parser.constraint_parser import TemplateConstraintParser


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def load_data():
    with open(os.path.join(DATA_DIR, "SCoRE2026_trainset.json"), "r", encoding="utf-8") as f:
        train = json.load(f)
    with open(os.path.join(DATA_DIR, "SCoRE2026_testset.json"), "r", encoding="utf-8") as f:
        test = json.load(f)
    return train, test


def run_symbolic_baseline(train, test):
    """运行Neuro-Symbolic基线（使用模板解析器）"""
    print("\n" + "=" * 60)
    print("基线测试: Neuro-Symbolic (模板解析器 + 符号求解器)")
    print("=" * 60)

    pipeline = SCOREPipeline(parser=TemplateConstraintParser())

    # 在训练集上评估（检查求解器覆盖度）
    print("\n--- 训练集评估（检查约束提取+求解覆盖度）---")
    results = pipeline.evaluate(train[:500])  # 先测试500条
    pipeline.print_stats()

    # 详细错误分析
    print("\n--- 错误样例分析（前10条）---")
    errors = [r for r in results["results"] if not r["correct"]]
    for err in errors[:10]:
        print(f"  {err['id']} ({err['domain']}): predicted={err['predicted']}, actual={err['actual']}")

    # 各领域多选/单选准确率
    print("\n--- 各领域详细分析 ---")
    for domain in ["time", "space", "social", "nature", "space+nature"]:
        domain_results = [r for r in results["results"] if r["domain"] == domain]
        if not domain_results:
            continue

        single_correct = sum(1 for r in domain_results
                           if r["correct"] and len(r["actual"]) == 1)
        multi_correct = sum(1 for r in domain_results
                          if r["correct"] and len(r["actual"]) > 1)
        single_total = sum(1 for r in domain_results if len(r["actual"]) == 1)
        multi_total = sum(1 for r in domain_results if len(r["actual"]) > 1)

        print(f"\n  {domain}:")
        print(f"    单选: {single_correct}/{single_total} "
              f"({single_correct/max(single_total,1)*100:.1f}%)")
        print(f"    多选: {multi_correct}/{multi_total} "
              f"({multi_correct/max(multi_total,1)*100:.1f}%)")

    return results


def generate_submission_file(test, output_name="submission_symbolic.json"):
    """生成测试集提交文件"""
    print("\n" + "=" * 60)
    print("生成测试集提交文件")
    print("=" * 60)

    pipeline = SCOREPipeline(parser=TemplateConstraintParser())
    output_path = os.path.join(OUTPUT_DIR, output_name)
    pipeline.generate_submission(test, output_path)


def analyze_solver_coverage(train):
    """分析求解器对训练集的覆盖度"""
    print("\n" + "=" * 60)
    print("求解器覆盖度分析")
    print("=" * 60)

    parser = TemplateConstraintParser()

    coverage = defaultdict(lambda: {"total": 0, "parsed": 0, "solved": 0, "verified": 0})

    for sample in train[:200]:  # 采样200条分析
        domain = sample["domain"]
        coverage[domain]["total"] += 1

        # 尝试解析
        constraints = parser.parse(sample["text"], domain, sample["question"], sample["language"])
        if constraints and (constraints.time or constraints.space or
                           constraints.social or constraints.nature):
            coverage[domain]["parsed"] += 1

    for domain, stats in sorted(coverage.items()):
        parse_rate = stats["parsed"] / max(stats["total"], 1) * 100
        print(f"  {domain:20s}: 解析率 {stats['parsed']:3d}/{stats['total']:3d} ({parse_rate:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="SCoRE2026 Baseline Testing")
    parser.add_argument("--mode", choices=["analyze", "baseline", "submit"],
                       default="analyze", help="运行模式")
    parser.add_argument("--samples", type=int, default=500,
                       help="评估使用的样本数")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train, test = load_data()

    if args.mode == "analyze":
        analyze_solver_coverage(train)
    elif args.mode == "baseline":
        run_symbolic_baseline(train, test)
    elif args.mode == "submit":
        generate_submission_file(test)
    else:
        # 全跑
        analyze_solver_coverage(train)
        run_symbolic_baseline(train, test)
        generate_submission_file(test)


if __name__ == "__main__":
    main()
