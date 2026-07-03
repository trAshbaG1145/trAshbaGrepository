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

## 项目结构

```
Project_NLP/
├── main.py                          # 主入口
├── requirements.txt                 # Python依赖
├── README.md                        # 本文件
├── SCoRE2026_项目执行方案.md         # 详细方案文档
├── data/
│   ├── SCoRE2026_trainset.json      # 训练集 (3600题)
│   └── SCoRE2026_testset.json       # 测试集 (1000题，无答案)
├── src/
│   ├── constraint_schema.py         # 统一约束Schema定义
│   ├── solvers/
│   │   ├── time_solver.py           # 时间推理求解器
│   │   ├── space_solver.py          # 空间推理求解器
│   │   ├── social_solver.py         # 社会关系求解器
│   │   ├── nature_solver.py         # 自然常识求解器
│   │   └── fusion_solver.py         # 融合域求解器
│   ├── parser/
│   │   └── constraint_parser.py     # 约束解析器 (LLM + 模板)
│   └── pipeline/
│       ├── score_pipeline.py        # 端到端Pipeline
│       └── answer_verifier.py       # 答案验证器
├── scripts/
│   ├── analyze_data.py              # 数据分析
│   ├── run_baseline.py              # 基线测试
│   ├── generate_fusion_data.py      # 融合域数据增强
│   ├── prepare_constraint_labels.py # 准备约束标注
│   └── train_constraint_parser.py   # LoRA微调脚本
├── outputs/                         # 输出目录
└── checkpoints/                     # 模型检查点
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据分析

```bash
python main.py analyze
```

### 3. CoT 训练数据生成（需 DeepSeek API Key）

```bash
# 先试100条
python main.py gen-cot --samples 100 --api_key sk-xxx
# 或设置环境变量
export DEEPSEEK_API_KEY=sk-xxx
python main.py gen-cot --samples 100
```

### 4. 模型微调（需 GPU）

```bash
python scripts/train_cot_model.py \
    --train_data outputs/cot_train_filtered.json \
    --output_dir checkpoints/cot_model
```

### 5. 测试集推理

```bash
python main.py infer-cot \
    --model_path checkpoints/cot_model/final \
    --output outputs/submission_cot.json
```

### 6. 基线测试（模板解析器，无需GPU）

```bash
python main.py baseline
```

## 技术方案

本方案采用 **CoT (Chain-of-Thought) 推理 + 知识蒸馏** 架构：

```
训练阶段:
  训练集 → [DeepSeek-V3 CoT推理] → 过滤(仅保留正确推理) → [LoRA微调 Qwen2.5-7B]

推理阶段:
  测试集 → [微调后 Qwen2.5-7B] → CoT推理链 + 答案JSON
```

- **DeepSeek-V3** 作为Teacher：在训练集上生成逐步推理链
- **Qwen2.5-7B** 作为Student：LoRA微调学习推理能力
- **推理链输出**：`## Reasoning` + `## Answer {"answers": ["A", "B"]}`

### 两种解析器

| 解析器 | 类名 | 说明 |
|--------|------|------|
| **模板解析器** | `TemplateConstraintParser` | 基于正则表达式的规则解析，无需LLM/GPU，作为基线 |
| **LLM解析器** | `LLMConstraintParser` | 使用微调后的LLM进行约束解析（主方案） |

## 关键挑战

| 挑战 | 说明 | 应对策略 |
|------|------|---------|
| 域偏移 | 训练97.2%单域 → 测试67.5%融合域 | 符号求解器天然跨域通用 |
| 多选 | 23.6%题目为不定项选择 | 答案验证器支持多选输出 |
| 双语 | 中文58% + 英文42% | 模板解析器中英分别处理 |
| 模型限制 | ≤8B参数 | Qwen2.5-7B-Instruct 推荐 |

## 基线性能（模板解析器，无需LLM）

在训练集前300条上的准确率：

| 领域 | 准确率 | 说明 |
|------|--------|------|
| time | ~12% | 时间线重建，支持中英双语和每周循环 |
| nature | ~6% | 属性匹配与约束传播 |
| space | ~2% | 空间布局回溯搜索 |
| social | ~0% | 亲属关系推导（最复杂） |
| 融合域 | ~0% | 跨域约束（模板解析器难以处理） |
| **总体** | **~5%** | 模板解析器基线 |

> ⚠️ 模板解析器的局限性：正则表达式无法覆盖所有自然语言表达方式。要达到有竞争力的准确率，需使用LLM约束解析器 + LoRA微调。

## 当前进度

### Neuro-Symbolic 路线（已放弃）
> DeepSeek-V3 约束提取正确率仅 3.6%，无法生成可用训练数据。

- [x] 项目结构搭建
- [x] 数据分析脚本
- [x] 约束Schema定义
- [x] 四大领域符号求解器
- [x] 融合域求解器
- [x] 端到端Pipeline
- [x] 答案验证器（支持填空/选择/选非三种题型）
- [x] 模板解析器基线
- [x] 基线测试框架
- [x] 约束标注Prompt生成
- [x] 批量约束标注脚本（batch_annotate.py）

### CoT 推理路线（当前主方案）
> DeepSeek API 生成 CoT 推理链 → LoRA 微调 Qwen2.5-7B → 端到端推理

- [x] CoT 数据生成脚本（generate_cot_data.py）
- [x] CoT 模型微调脚本（train_cot_model.py）
- [x] CoT 推理脚本（run_cot_inference.py）
- [ ] CoT 训练数据生成（需 DeepSeek API）
- [ ] 模型微调（需 GPU）
- [ ] 最终推理与提交
- [x] LoRA微调脚本框架
- [ ] 约束标注数据生成（需LLM辅助，prompt已就绪）
- [ ] 模型训练（需GPU环境）
- [ ] 最终调优与提交

## 下一步

1. **使用LLM生成约束标注**：`outputs/annotation_prompts.json` 已包含250条标注prompt，用强LLM（GPT-4/Claude/DeepSeek）批量生成约束JSON
2. **LoRA微调约束解析器**：`python scripts/train_constraint_parser.py`
3. **集成LLM解析器**：将微调后的模型接入Pipeline替换模板解析器
4. **消融实验**：对比纯LLM CoT vs Neuro-Symbolic方案

### 已生成的输出文件

| 文件 | 说明 |
|------|------|
| `outputs/submission.json` | 测试集1000条提交（252条非空预测） |
| `outputs/fusion_augmented_train.json` | 200条合成融合域训练数据 |
| `outputs/combined_train.json` | 3800条合并训练集 |
| `outputs/annotation_prompts.json` | 250条LLM约束标注prompt |
