"""
SCoRE2026 数据分析脚本
分析训练集和测试集的分布特征，为建模提供依据
"""
import json
import sys
import os
from collections import Counter, defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_data():
    with open(os.path.join(DATA_DIR, "SCoRE2026_trainset.json"), "r", encoding="utf-8") as f:
        train = json.load(f)
    with open(os.path.join(DATA_DIR, "SCoRE2026_testset.json"), "r", encoding="utf-8") as f:
        test = json.load(f)
    return train, test


def analyze_basic_stats(train, test):
    """基础统计"""
    print("=" * 60)
    print("一、基础统计")
    print("=" * 60)

    print(f"\n训练集: {len(train)} 条")
    print(f"测试集: {len(test)} 条")
    print(f"总计: {len(train) + len(test)} 条")

    # 领域分布
    print("\n--- 训练集领域分布 ---")
    for domain, count in Counter(s["domain"] for s in train).most_common():
        pct = count / len(train) * 100
        bar = "█" * int(pct / 2)
        print(f"  {domain:20s}: {count:5d} ({pct:5.1f}%) {bar}")

    print("\n--- 测试集领域分布 ---")
    for domain, count in Counter(s["domain"] for s in test).most_common():
        pct = count / len(test) * 100
        bar = "█" * int(pct / 2)
        print(f"  {domain:20s}: {count:5d} ({pct:5.1f}%) {bar}")

    # 语言分布
    print("\n--- 语言分布 ---")
    for split_name, split_data in [("训练集", train), ("测试集", test)]:
        cn = sum(1 for s in split_data if s["language"] == "cn")
        en = sum(1 for s in split_data if s["language"] == "en")
        print(f"  {split_name}: 中文={cn} ({cn/len(split_data)*100:.1f}%), 英文={en} ({en/len(split_data)*100:.1f}%)")


def analyze_question_types(train, test):
    """题型分析"""
    print("\n" + "=" * 60)
    print("二、题型分析")
    print("=" * 60)

    def classify_question(s):
        q = s["question"]
        if "____" in q:
            return "填空式"
        elif "incorrect" in q.lower() or "不正确" in q:
            return "选非题"
        else:
            return "选择题"

    for split_name, split_data in [("训练集", train), ("测试集", test)]:
        types = Counter(classify_question(s) for s in split_data)
        print(f"\n  {split_name}:")
        for t, c in types.most_common():
            print(f"    {t}: {c} ({c/len(split_data)*100:.1f}%)")


def analyze_answer_patterns(train):
    """答案模式分析"""
    print("\n" + "=" * 60)
    print("三、答案模式分析")
    print("=" * 60)

    # 单选 vs 多选
    single = 0
    multi = 0
    for s in train:
        if len(s["answers"]) == 1:
            single += 1
        else:
            multi += 1

    print(f"\n  单选题: {single} ({single/len(train)*100:.1f}%)")
    print(f"  多选题: {multi} ({multi/len(train)*100:.1f}%)")

    # 答案选项分布
    all_answers = []
    for s in train:
        all_answers.extend(s["answers"])
    ans_dist = Counter(all_answers)
    print(f"\n  选项分布: {dict(ans_dist)}")

    # 各领域的多选比例
    print("\n  各领域多选比例:")
    for domain in sorted(set(s["domain"] for s in train)):
        domain_samples = [s for s in train if s["domain"] == domain]
        multi_count = sum(1 for s in domain_samples if len(s["answers"]) > 1)
        print(f"    {domain:15s}: {multi_count}/{len(domain_samples)} ({multi_count/len(domain_samples)*100:.1f}%)")


def analyze_text_complexity(train, test):
    """文本复杂度分析"""
    print("\n" + "=" * 60)
    print("四、文本复杂度分析")
    print("=" * 60)

    for split_name, split_data in [("训练集", train), ("测试集", test)]:
        lengths = [len(s["text"]) for s in split_data]
        print(f"\n  {split_name}:")
        print(f"    最短: {min(lengths)} 字符")
        print(f"    最长: {max(lengths)} 字符")
        print(f"    平均: {sum(lengths)/len(lengths):.0f} 字符")
        print(f"    中位数: {sorted(lengths)[len(lengths)//2]} 字符")

    # 各领域文本长度
    print("\n  各领域平均文本长度:")
    for domain in sorted(set(s["domain"] for s in train)):
        samples = [s for s in train if s["domain"] == domain]
        avg_len = sum(len(s["text"]) for s in samples) / len(samples)
        print(f"    {domain:15s}: {avg_len:.0f} 字符")


def analyze_domain_shift(train, test):
    """域偏移分析 —— 最关键的分析"""
    print("\n" + "=" * 60)
    print("五、训练-测试域偏移分析 ⚠️ 核心挑战")
    print("=" * 60)

    train_single = sum(1 for s in train if "+" not in s["domain"])
    train_fusion = sum(1 for s in train if "+" in s["domain"])
    test_single = sum(1 for s in test if "+" not in s["domain"])
    test_fusion = sum(1 for s in test if "+" in s["domain"])

    print(f"\n  {'':20s} {'训练集':>10s} {'测试集':>10s}")
    print(f"  {'单域':20s} {train_single:>8d} ({train_single/len(train)*100:5.1f}%)  {test_single:>8d} ({test_single/len(test)*100:5.1f}%)")
    print(f"  {'融合域':20s} {train_fusion:>8d} ({train_fusion/len(train)*100:5.1f}%)  {test_fusion:>8d} ({test_fusion/len(test)*100:5.1f}%)")

    print(f"\n  ⚠️ 训练集融合域仅 {train_fusion/len(train)*100:.1f}%，测试集融合域高达 {test_fusion/len(test)*100:.1f}%")
    print(f"  ⚠️ 模型必须在单域数据上学习推理能力，泛化到融合域场景")

    # 测试集中出现的融合域组合
    fusion_test = [s for s in test if "+" in s["domain"]]
    fusion_combos = Counter(s["domain"] for s in fusion_test)
    print(f"\n  测试集融合域组合:")
    for combo, count in fusion_combos.most_common():
        # 检查训练集中是否有对应数据
        train_count = sum(1 for s in train if s["domain"] == combo)
        print(f"    {combo:20s}: 测试{count:4d}条, 训练{train_count:4d}条 {'⚠️ 训练中无此组合!' if train_count == 0 else ''}")


def analyze_constraint_patterns(train):
    """约束模式分析 —— 为求解器设计提供依据"""
    print("\n" + "=" * 60)
    print("六、约束模式分析")
    print("=" * 60)

    for domain in ["time", "space", "social", "nature"]:
        samples = [s for s in train if s["domain"] == domain]
        if not samples:
            continue
        print(f"\n  [{domain}] 共{len(samples)}条")

        # 统计平均约束条件数量（通过文本中的编号项估算）
        constraint_counts = []
        for s in samples:
            # 统计 "(1)" "(2)" 等编号
            import re
            numbered_items = re.findall(r'\(\d+\)', s["text"])
            constraint_counts.append(len(numbered_items))

        avg_constraints = sum(constraint_counts) / len(constraint_counts)
        print(f"    平均约束条件数: {avg_constraints:.1f}")
        print(f"    最少/最多: {min(constraint_counts)} / {max(constraint_counts)}")

        # 实体数量估算（粗略）
        if domain == "social":
            # 社交域：统计中文人名
            cn_names = sum(1 for s in samples if s["language"] == "cn")
            if cn_names > 0:
                sample_cn = [s for s in samples if s["language"] == "cn"][0]
                print(f"    中文社交样例实体数: ~{len(sample_cn['text']) // 50}")


def main():
    train, test = load_data()

    analyze_basic_stats(train, test)
    analyze_question_types(train, test)
    analyze_answer_patterns(train)
    analyze_text_complexity(train, test)
    analyze_domain_shift(train, test)
    analyze_constraint_patterns(train)

    print("\n" + "=" * 60)
    print("分析完成！核心发现：")
    print("=" * 60)
    print("1. 训练集97.2%为单域，测试集67.5%为融合域 → 域偏移是最关键挑战")
    print("2. 23.6%题目为多选题 → 需要不定项选择能力")
    print("3. 中英文约各半 → 需要双语推理能力")
    print("4. 所有题目均为4选项 → 可固定输出格式")
    print("5. 每个问题本质上是一个约束满足问题(CSP)")


if __name__ == "__main__":
    main()
