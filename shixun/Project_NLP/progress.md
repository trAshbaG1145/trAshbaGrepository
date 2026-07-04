# SCoRE2026 项目进度报告

> **项目：** CCL 2026 任务十 —— 基于情景的常识推理评测（SCoRE）
> **主办方：** 北京大学 + 华为 | **难度等级：** B级（中高难度）
> **团队：** trAshbaG 小组

---

## Day 1（7月3日）—— 方案设计、环境搭建、基础架构

### 完成内容

1. **任务分析**
   - 研读 SCoRE2026 任务说明，识别五大推理类型
   - 核心挑战分析：训练-测试域偏移（97.2%单域→67.5%融合域）、中英双语、23.6%多选
   - 竞赛约束：模型≤8B、禁止外部数据、测试集不可训练

2. **技术方案设计**
   - 初始路线：Neuro-Symbolic 混合系统（LLM 约束解析 + 符号求解器）
   - 选定 Qwen2.5-7B-Instruct 主模型
   - 编写 `SCoRE2026_项目执行方案.md`

3. **环境搭建与基础代码**
   - Python 深度学习环境、项目目录结构
   - `analyze_data.py`：训练集 3600 条数据分析
   - `constraint_schema.py`：五大领域统一约束 Schema

### 产出
| 文件 | 说明 |
|------|------|
| `SCoRE2026_项目执行方案.md` | 详细技术方案 |
| `scripts/analyze_data.py` | 数据分析脚本 |
| `src/constraint_schema.py` | 约束 Schema 定义 |

---

## Day 2（7月4日）—— 路线验证、策略转向、CoT 实现

### 上午：Neuro-Symbolic 路线开发与验证

1. **符号求解器开发**
   - `time_solver.py`：约束传播 + 拓扑排序 + 周循环
   - `space_solver.py`：CSP 回溯搜索，3×2 网格布局
   - `social_solver.py`：关系图 + 中文亲属称谓知识库
   - `nature_solver.py`：属性矩阵 + 排除法
   - `fusion_solver.py`：分阶段求解 + 跨域信息注入

2. **端到端 Pipeline**
   - `score_pipeline.py`：约束解析→求解→验证
   - `answer_verifier.py`：填空/选择正确/选择错误三种题型

3. **模板解析器基线**
   - 正则表达式提取约束，总体准确率 ~5%
   - 结论：正则无法覆盖自然语言多样性

4. **LLM 约束提取验证 —— 关键发现**
   - `batch_annotate.py`：批量调用 DeepSeek API 生成约束 JSON
   - DeepSeek-V3 约束提取正确率：**3.6%**（250 条测试）
   - DeepSeek-R1 更差：50% JSON 提取失败
   - **决策：放弃 Neuro-Symbolic 路线**

### 下午：CoT 推理路线实现

1. **CoT 数据生成** (`generate_cot_data.py`)
   - DeepSeek-V3 API 批量生成 CoT 推理链
   - 格式：`## Reasoning` + `## Answer {"answers": ["A","B"]}`
   - 质量过滤：仅保留预测正确的推理链
   - 断点续传支持（API 缓存）
   - **全量生成结果**：3600 条 → 2668 条过滤后（74.1% 正确率）

   | 领域 | 正确率 | 训练样本 |
   |------|--------|---------|
   | nature | 87.5% | 875 |
   | time | 86.1% | 861 |
   | social | 63.4% | 317 |
   | space | 56.2% | 562 |
   | space+nature | 53.0% | 53 |

2. **LoRA 微调脚本** (`train_cot_model.py`)
   - Qwen2.5-7B-Instruct + ChatML 格式
   - LoRA config: r=16, alpha=32

3. **推理脚本** (`run_cot_inference.py`)
   - PeftModel 加载 + 批量推理
   - 多策略答案提取（JSON→regex→回退）

4. **云平台一键训练** (`run_train.py`)
   - 自动环境检查 + GPU 诊断
   - ModelScope/HF 镜像双通道模型下载
   - CUDA 兼容性自动处理（torchvision/torchaudio 冲突修复）
   - bf16/fp16 自适应
   - 训练+推理完整流程

5. **Bug 修复（10+ 个）**
   - 推理 prompt 与训练不一致
   - pip shell 重定向符转义
   - bf16 旧 GPU 兼容
   - device_map + .to() 冲突
   - CUDA 不可用时崩溃
   - pip -q 静默错误
   - RLock 死锁
   - torchvision/torchaudio CUDA 版本冲突
   - ModelScope 路径名转义

### 产出
| 文件 | 说明 |
|------|------|
| `src/solvers/` | 5 个符号求解器 |
| `src/pipeline/` | Pipeline + 答案验证器 |
| `scripts/batch_annotate.py` | LLM 约束标注（N-S 验证用） |
| `scripts/generate_cot_data.py` | CoT 数据生成 |
| `scripts/train_cot_model.py` | LoRA 微调 |
| `scripts/run_cot_inference.py` | 推理+提交 |
| `run_train.py` | 云平台一键训练 |
| `outputs/cot_train_filtered.json` | 2668 条 CoT 训练数据 |
| `docs/plans/` | CoT 实施计划 |

---

## 后续计划

| 阶段 | 内容 | 状态 |
|------|------|------|
| GPU 训练 | Qwen2.5-7B LoRA 微调 (3 epochs) | 🔄 进行中 |
| 推理提交 | 测试集 1000 条推理 → submission_cot.json | ⏳ 待训练完成 |
| 消融实验 | 零样本 vs CoT微调 vs 模板基线 | ⏳ 待推理完成 |
| 技术报告 | 4 页 CCL 格式 | ⏳ 待实验完成 |
| 答辩准备 | PPT + Demo | ⏳ 待报告完成 |

### 当前系统架构

```
训练阶段:
  训练集(3600) → DeepSeek-V3 CoT推理 → 过滤(74.1%) → 2668条训练数据
  → LoRA微调 Qwen2.5-7B → checkpoints/cot_model/

推理阶段:
  测试集(1000) → 微调后 Qwen2.5-7B → ## Reasoning + ## Answer
  → 答案提取 → [{"id": "...", "answers": ["A","B"]}]
```

### 路线决策记录

```
Neuro-Symbolic (Day 1-2)              CoT Reasoning (Day 2+)
┌────────────────────────┐            ┌──────────────────────────┐
│ NL → 约束JSON → 符号求解│  ──放弃──→ │ NL → CoT推理链 → 答案JSON │
│ DeepSeek 提取正确率 3.6%│            │ Teacher 零样本 74.1%      │
└────────────────────────┘            └──────────────────────────┘
```

核心教训：精确结构化提取是 LLM 的弱项，端到端推理是 LLM 的强项。

---

> 📅 创建日期：2026年7月3日 | 📝 最后更新：2026年7月4日
