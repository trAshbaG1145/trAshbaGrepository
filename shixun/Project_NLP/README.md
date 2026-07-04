# SCoRE2026 - 基于情景的常识推理评测

> CCL 2026 任务十 | 难度 B级（中高难度）
> 主办方：北京大学 + 华为

## 项目概述

评估大语言模型在常识场景下的复杂逻辑推理能力，涵盖五大推理类型：
- ⏰ **时间推理** - 事件时间线重建
- 📐 **空间推理** - 多维空间布局推理
- 👥 **社会推理** - 亲属/社交关系网络
- 🌿 **自然推理** - 物体属性与分类
- 🔀 **融合推理** - 跨域联合约束求解

## 技术方案

采用 **CoT (Chain-of-Thought) 推理 + 知识蒸馏** 架构：

```
训练阶段:
  训练集 → [DeepSeek-V3 CoT推理] → 过滤(仅保留正确推理) → [LoRA微调 Qwen2.5-7B]

推理阶段:
  测试集 → [微调后 Qwen2.5-7B] → CoT推理链 + 答案JSON
```

- **DeepSeek-V3** 作为 Teacher：在训练集上生成逐步推理链（零样本正确率 74.1%）
- **Qwen2.5-7B** 作为 Student：LoRA 微调学习推理能力（满足 ≤8B 约束）
- **推理链格式**：`## Reasoning` + `## Answer {"answers": ["A", "B"]}`

### 路线演进

| 路线 | 思路 | 正确率 | 状态 |
|------|------|--------|------|
| Neuro-Symbolic | LLM 提取约束 JSON → 符号求解器 → 答案 | 3.6% | ❌ 已放弃 |
| **CoT 推理** | LLM 端到端逐步推理 → 直接输出答案 | **74.1%** (Teacher) | ✅ 当前方案 |

Neuro-Symbolic 失败原因：DeepSeek-V3 无法可靠地将自然语言转化为精确结构化约束。CoT 路线避免了中间表示的精度损失。

## 项目结构

```
Project_NLP/
├── run_train.py                      # 云平台一键训练脚本
├── main.py                           # 主入口
├── requirements.txt                  # Python依赖
├── .gitignore
│
├── data/
│   ├── SCoRE2026_trainset.json       # 训练集 (3600题)
│   └── SCoRE2026_testset.json        # 测试集 (1000题，无答案)
│
├── outputs/
│   └── cot_train_filtered.json       # CoT训练数据 (2668条, 74.1%正确)
│
├── scripts/
│   ├── generate_cot_data.py          # CoT数据生成 (DeepSeek API)
│   ├── train_cot_model.py            # LoRA微调脚本
│   ├── run_cot_inference.py          # 推理+提交生成
│   ├── analyze_data.py               # 数据分析
│   ├── run_baseline.py               # 模板解析器基线 (~5%)
│   └── (batch_annotate.py, train_constraint_parser.py 等 # N-S路线保留)
│
├── src/
│   ├── cot/                          # CoT推理模块
│   ├── solvers/                      # 符号求解器 (保留作消融对比)
│   ├── parser/                       # 约束解析器 (保留)
│   └── pipeline/                     # Pipeline (保留)
│
└── docs/
    ├── SCoRE2026.pdf                 # 任务说明
    └── plans/                        # 实施计划
```

## 快速开始

### 1. CoT 训练数据生成（本地，需 DeepSeek API）

```bash
pip install openai
export DEEPSEEK_API_KEY=sk-xxx
python main.py gen-cot --samples 100   # 试跑
python main.py gen-cot                 # 全量3600条
```

### 2. 模型训练 + 推理（云 GPU 一键运行）

```bash
python run_train.py --modelscope       # 国内云平台
python run_train.py                    # 海外云平台
```

### 3. 基线测试（本地，无需 GPU）

```bash
python main.py baseline               # 模板解析器基线
```

## 关键挑战

| 挑战 | 说明 | 应对策略 |
|------|------|---------|
| 域偏移 | 训练 97.2% 单域 → 测试 67.5% 融合域 | CoT 推理链天然跨域泛化 |
| 多选 | 23.6% 题目为不定项选择 | 输出格式 `{"answers": ["A", "B"]}` |
| 双语 | 中文 58% + 英文 42% | 统一 CoT prompt，不区分语言 |
| 模型限制 | ≤8B 参数 | Qwen2.5-7B-Instruct |

## 当前进度

### CoT 推理路线

- [x] CoT 数据生成脚本
- [x] CoT 数据全量生成（3600→2668 条，74.1% 正确率）
- [x] LoRA 微调脚本 + 云平台一键训练脚本
- [x] CoT 推理 + 提交生成脚本
- [x] 文档更新（方案、进度、计划）
- [ ] 模型微调（云平台进行中）
- [ ] 测试集推理与提交
- [ ] 消融实验（CoT vs 模板 vs 零样本）
- [ ] 技术报告（4 页 CCL 格式）

### Neuro-Symbolic 路线（已放弃，代码保留）

- [x] 约束 Schema + 5 领域符号求解器
- [x] 模板解析器基线（~5%）
- [x] 约束标注 pipeline
- [x] 验证 LLM 约束提取不可行（DeepSeek 仅 3.6%）

## CoT 训练数据质量

| 领域 | 正确率 | 训练样本 |
|------|--------|---------|
| nature | 87.5% | 875 |
| time | 86.1% | 861 |
| social | 63.4% | 317 |
| space | 56.2% | 562 |
| space+nature | 53.0% | 53 |
| **总计** | **74.1%** | **2668** |
