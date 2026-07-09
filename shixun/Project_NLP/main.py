"""
SCoRE2026 项目主入口

用法（无需LLM/GPU/API Key）：
    python main.py analyze               数据分析
    python main.py baseline              基线测试
    python main.py submit                生成提交文件
    python main.py interactive           交互式调试
"""
import sys
import os
import argparse
import json
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def load_data():
    with open(os.path.join(DATA_DIR, "SCoRE2026_trainset.json"), "r", encoding="utf-8") as f:
        train = json.load(f)
    with open(os.path.join(DATA_DIR, "SCoRE2026_testset.json"), "r", encoding="utf-8") as f:
        test = json.load(f)
    return train, test


def cmd_analyze():
    from scripts.analyze_data import main as m
    m()


def cmd_baseline(samples=300):
    """基线测试：模板解析器 + 符号求解器（无需LLM）"""
    print(f"Baseline: template parser + symbolic solver ({samples} samples)")
    from src.pipeline.score_pipeline import SCOREPipeline
    from src.parser.constraint_parser import TemplateConstraintParser

    train, _ = load_data()
    pipeline = SCOREPipeline(parser=TemplateConstraintParser())

    eval_samples = train[:min(samples, len(train))]
    results = pipeline.evaluate(eval_samples)
    pipeline.print_stats()

    # 错误样例
    errors = [r for r in results["results"] if not r["correct"]]
    print(f"\n--- Errors ({len(errors)}/{len(eval_samples)}) ---")
    for e in errors[:15]:
        print(f"  {e['id']} [{e['domain']}] pred={e['predicted']} actual={e['actual']}")

    # 分题型统计
    from collections import defaultdict
    qtype_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r, s in zip(results["results"], eval_samples):
        q = s["question"]
        if "____" in q:
            qt = "fill"
        elif "不正确" in q or "incorrect" in q.lower():
            qt = "incorrect"
        else:
            qt = "select"
        qtype_stats[qt]["total"] += 1
        if r["correct"]:
            qtype_stats[qt]["correct"] += 1
    print("\n--- By question type ---")
    for qt, st in sorted(qtype_stats.items()):
        acc = st["correct"] / max(st["total"], 1) * 100
        print(f"  {qt:10s}: {st['correct']:3d}/{st['total']:3d} ({acc:.0f}%)")

    return results


def cmd_submit():
    """生成测试集提交文件（纯规则引擎）"""
    print("Generating submission (template parser)...")
    from src.pipeline.score_pipeline import SCOREPipeline
    from src.parser.constraint_parser import TemplateConstraintParser

    _, test = load_data()
    pipeline = SCOREPipeline(parser=TemplateConstraintParser())
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "submission.json")
    pipeline.generate_submission(test, output_path)


def cmd_augment():
    from scripts.generate_fusion_data import main as m
    m()


def cmd_prepare_labels():
    from scripts.prepare_constraint_labels import main as m
    m()


def cmd_interactive():
    """交互式单题测试"""
    from src.pipeline.score_pipeline import SCOREPipeline
    from src.parser.constraint_parser import TemplateConstraintParser

    print("\nSCoRE2026 Interactive Solver (type 'quit' to exit)\n")
    pipeline = SCOREPipeline(parser=TemplateConstraintParser())

    while True:
        print("--- New Problem ---")
        domain = input("Domain (time/space/social/nature): ").strip()
        if domain.lower() == "quit":
            break
        text = input("Text: ").strip()
        if text.lower() == "quit":
            break
        question = input("Question: ").strip()
        options = {}
        for i in range(4):
            letter = chr(ord("A") + i)
            opt = input(f"Option {letter}: ").strip()
            if not opt:
                break
            options[letter] = opt

        sample = {"id": "test", "domain": domain, "language": "cn",
                  "text": text, "question": question, "options": options}
        answers = pipeline.solve_single(sample)
        print(f"  => Answer(s): {answers}\n")


def cmd_batch_annotate():
    """批量约束标注生成（调用LLM API）"""
    from scripts.batch_annotate import main as m
    m()


def cmd_gen_cot():
    """CoT推理数据生成（DeepSeek API）"""
    from scripts.generate_cot_data import main as m
    m()


def cmd_infer_cot():
    """CoT模型推理（需GPU）"""
    from scripts.run_cot_inference import main as m
    m()


def cmd_full():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cmd_analyze()
    cmd_baseline(samples=500)
    cmd_submit()
    print(f"\nDone. Outputs in {OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="SCoRE2026 Pipeline")
    parser.add_argument("command", nargs="?", default="analyze",
                       choices=["analyze", "baseline", "submit", "augment",
                               "prepare-labels", "batch-annotate",
                               "gen-cot", "train-cot", "infer-cot",
                               "full", "interactive"])
    args, unknown = parser.parse_known_args()

    cmds = {
        "analyze": cmd_analyze,
        "baseline": lambda: cmd_baseline(samples=300),
        "submit": cmd_submit,
        "augment": cmd_augment,
        "prepare-labels": cmd_prepare_labels,
        "batch-annotate": cmd_batch_annotate,
        "gen-cot": cmd_gen_cot,
        "infer-cot": cmd_infer_cot,
        "full": cmd_full,
        "interactive": cmd_interactive,
    }

    fn = cmds.get(args.command)
    if fn:
        # Pass through extra args to sub-command scripts
        sys.argv = [sys.argv[0]] + unknown
        fn()


if __name__ == "__main__":
    main()
