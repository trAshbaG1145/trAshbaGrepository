# SCoRE2026 项目进度报告

> **项目：** CCL 2026 任务十 —— 基于情景的常识推理评测（SCoRE）
> **主办方：** 北京大学 + 华为 | **难度等级：** B级（中高难度）
> **团队：** trAshbaG 小组

---

## Day 1（7月3日）—— 任务讨论、小组分工、环境搭建

### 完成内容

1. **任务分析与讨论**
   - 深入研读 SCoRE2026 任务说明，明确评测目标：评估大语言模型在常识场景下的复杂逻辑推理能力
   - 识别五大推理类型：时间推理、空间推理、社会推理、自然推理、融合推理
   - 分析核心挑战：(1) 训练-测试域偏移（训练集 97.2% 单域 → 测试集 67.5% 融合域）；(2) 多步约束求解；(3) 中英双语推理；(4) 23.6% 题目为不定项选择
   - 明确竞赛约束：模型规模 ≤ 8B（Dense）、禁止使用外部数据集、测试集不可用于训练

2. **小组分工**
   - 确定采用 Neuro-Symbolic 混合系统作为初始技术路线：LLM 负责约束解析，符号求解器负责逻辑推理
   - 分工安排：数据分析和 Schema 设计、约束解析器开发、符号求解器实现、Pipeline 集成与验证
   - 选定 Qwen2.5-7B-Instruct 作为主模型（中英双语能力强，开源生态好）

3. **环境搭建**
   - 配置 Python 深度学习环境：PyTorch 2.1+, Transformers 4.40+, PEFT, bitsandbytes
   - 安装 LLM API 客户端（OpenAI SDK → DeepSeek API, Anthropic SDK）
   - 搭建项目目录结构：`src/solvers/`（求解器）、`src/parser/`（解析器）、`src/pipeline/`（Pipeline）、`scripts/`（工具脚本）
   - 初始化 Git 仓库，编写 `requirements.txt`

4. **数据分析初步**
   - 编写 `scripts/analyze_data.py`，对训练集 3600 条数据进行统计分析
   - 确认数据格式：每条包含 id、domain、language、text、question、options、answers
   - 统计领域分布、中英文比例、多答案题目占比等基础指标

### 产出文件
| 文件 | 说明 |
|------|------|
| `requirements.txt` | Python 依赖清单 |
| `README.md` | 项目说明文档 |
| `SCoRE2026_项目执行方案.md` | 详细技术方案 |
| `scripts/analyze_data.py` | 数据分析脚本 |
| `src/constraint_schema.py` | 统一约束 Schema 定义 |

---

## Day 2（7月4日）—— 基准模型实现（一）：约束 Schema 与符号求解器

### 完成内容

1. **统一约束 Schema 设计**
   - 为五大领域定义标准化 JSON 约束格式
   - 时间约束：支持绝对时间（"星期三"）和相对时间（"after/before + offset"），处理每周 7 天循环
   - 空间约束：支持网格布局（above/below/adjacent/position），行列坐标系
   - 社会约束：亲属关系（kinship）、职场关系，支持中文复杂称谓
   - 自然约束：属性匹配（property_match）、类别约束（category）、排除法推理
   - 融合约束：跨域联合表示，分阶段求解接口

2. **符号求解器开发**
   - `time_solver.py`：约束传播 + 拓扑排序，将相对时间约束转化为绝对时间线，模 7 运算处理周期性
   - `space_solver.py`：CSP 回溯搜索，构建网格布局，逐条应用空间约束，搜索所有可行解
   - `social_solver.py`：关系图构建 + 传递闭包推导，支持中文亲属称谓（如"姐夫"、"岳母"、"儿媳"）
   - `nature_solver.py`：属性矩阵 + 约束传播 + 排除法，逐步锁定实体-属性对应关系
   - `fusion_solver.py`：分阶段求解，先求解一个领域再注入另一个领域

3. **端到端 Pipeline 搭建**
   - `score_pipeline.py`：串联约束解析 → 符号求解 → 答案验证的完整流程
   - `answer_verifier.py`：支持填空（fill_blank）、选择正确（select_correct）、选择错误（select_incorrect）三种题型
   - 支持多答案输出（不定项选择）

### 产出文件
| 文件 | 说明 |
|------|------|
| `src/constraint_schema.py` | 完善约束 Schema 定义 |
| `src/solvers/time_solver.py` | 时间推理求解器 |
| `src/solvers/space_solver.py` | 空间推理求解器 |
| `src/solvers/social_solver.py` | 社会关系求解器 |
| `src/solvers/nature_solver.py` | 自然常识求解器 |
| `src/solvers/fusion_solver.py` | 融合域求解器 |
| `src/pipeline/score_pipeline.py` | 端到端 Pipeline |
| `src/pipeline/answer_verifier.py` | 答案验证器 |

---

## Day 3（7月5日）—— 基准模型实现（二）：模板解析器与基线测试

### 完成内容

1. **模板约束解析器**
   - `TemplateConstraintParser`：基于正则表达式从自然语言中提取约束条件
   - 支持中英文双语解析（中文 58% + 英文 42%）
   - 覆盖常见的自然语言表达模式：时间表达（"星期三"、"the day after"）、空间表达（"上方"、"to the left of"）、亲属称谓（"XX的爸爸"、"XX's mother"）

2. **基线测试框架**
   - 编写 `scripts/run_baseline.py`，在训练集前 300 条上评估模板解析器性能
   - 基线结果：
     - 时间推理：~12%（时间线重建较好，但正则覆盖不全）
     - 自然推理：~6%（属性匹配有一定效果）
     - 空间推理：~2%（回溯搜索开销大，约束提取困难）
     - 社会推理：~0%（中文亲属关系过于复杂，正则无法处理）
     - 融合域：~0%（跨域约束超出模板解析器能力）
     - **总体准确率：~5%**
   - 结论：模板解析器只能作为基线，需要 LLM 辅助约束解析

3. **约束标注数据准备**
   - 编写 `scripts/prepare_constraint_labels.py`：生成 250 条 LLM 约束标注 prompt
   - 编写 `scripts/batch_annotate.py`：批量调用强 LLM（GPT-4/Claude/DeepSeek）生成约束 JSON 标注
   - 编写 `scripts/train_constraint_parser.py`：LoRA 微调框架（为 Neuro-Symbolic 路线准备）

4. **Neuro-Symbolic 路线评估与决策**
   - 经测试，DeepSeek-V3 约束提取正确率仅 3.6%，无法生成可用训练数据
   - **重大决策：放弃 Neuro-Symbolic 路线，转向 CoT 推理路线**
   - 新方案：DeepSeek-V3 生成 CoT 推理链 → 按正确答案过滤 → LoRA 微调 Qwen2.5-7B → 端到端推理输出

### 产出文件
| 文件 | 说明 |
|------|------|
| `src/parser/constraint_parser.py` | 约束解析器（模板 + LLM） |
| `scripts/run_baseline.py` | 基线测试脚本 |
| `scripts/prepare_constraint_labels.py` | 约束标注准备 |
| `scripts/batch_annotate.py` | 批量 LLM 标注 |
| `scripts/train_constraint_parser.py` | LoRA 微调脚本 |
| `outputs/submission.json` | 测试集基线提交（252 条非空预测） |
| `outputs/fusion_augmented_train.json` | 200 条合成融合域数据 |
| `outputs/annotation_prompts.json` | 250 条约束标注 prompt |

### 路线调整说明
Neuro-Symbolic 路线的核心瓶颈在于：强 LLM 无法可靠地从自然语言中提取结构化约束（正确率仅 3.6%）。相比之下，CoT 路线让 LLM 直接从自然语言端到端推理，避免了中间结构化表示的准确率损失。原有 Neuro-Symbolic 代码保留作为消融实验对比基线。

---

## Day 4（7月6日）—— 模型改进（一）：CoT 数据生成与训练框架

### 完成内容

1. **CoT 推理数据生成脚本**
   - 编写 `scripts/generate_cot_data.py`：通过 DeepSeek API 在训练集上生成 CoT 推理链
   - CoT Prompt 设计：
     - System prompt：要求模型按 `## Reasoning` + `## Answer {"answers": ["A", "B"]}` 格式输出
     - 明确多选支持、中英双语、三种题型（填空/选择正确/选择错误）
   - 答案提取：实现多种解析策略（JSON 正则、独立 JSON、纯文本回退）
   - 并发调用：ThreadPoolExecutor 支持并发 API 请求，提升数据生成效率
   - 质量过滤：仅保留预测正确的推理链作为训练数据（quality_filter=True）
   - 统计输出：分领域统计正确率，指导后续数据策略

2. **CoT 模型 LoRA 微调脚本**
   - 编写 `scripts/train_cot_model.py`：
     - 支持 Qwen2.5-7B-Instruct 的 ChatML 格式输入
     - LoRA 配置：r=16, alpha=32, target_modules 覆盖所有 attention 和 FFN 线性层
     - 训练参数：3 epochs, batch_size=4, gradient_accumulation=4, learning_rate=2e-4
     - 支持 wandb 日志记录，best model 自动保存
     - 90/10 训练/验证集分割

3. **CoT 推理脚本**
   - 编写 `scripts/run_cot_inference.py`：
     - 加载 LoRA 微调后的模型（PeftModel）
     - 支持 vLLM 批量推理（加速）和 Transformers 逐条推理（兼容性好）
     - 多策略答案提取（JSON → regex → 纯文本回退）
     - 生成标准提交格式 `[{"id": "...", "answers": ["A", "B"]}]`

4. **项目入口集成**
   - 更新 `main.py`：添加 `gen-cot`、`train-cot`、`infer-cot` 三个子命令
   - 创建 `src/cot/__init__.py` CoT 推理模块

### 产出文件
| 文件 | 说明 |
|------|------|
| `scripts/generate_cot_data.py` | CoT 训练数据生成 |
| `scripts/train_cot_model.py` | CoT 模型 LoRA 微调 |
| `scripts/run_cot_inference.py` | CoT 模型推理 |
| `src/cot/__init__.py` | CoT 模块定义 |
| `main.py` | 更新：新增 CoT 子命令 |

---

## Day 5（7月7日）—— 模型改进（二）：数据增强与系统优化

### 完成内容

1. **融合域数据增强**
   - 编写 `scripts/generate_fusion_data.py`：程序化生成融合域训练数据
   - 策略：利用单域模板组合生成跨域问题（如 social+time、space+nature）
   - 生成 200 条合成融合域数据，缓解训练集中融合域仅占 2.8% 的严重不平衡
   - 合并为 3800 条训练集（`outputs/combined_train.json`）

2. **CoT 数据质量优化**
   - 针对低准确率领域（social、fusion）优化 prompt 模板
   - 增加推理链完整性检查：推理链必须包含明确的实体识别、约束列举、逐步推导过程
   - 空洞答案检测：过滤掉无推理过程直接输出答案的样本

3. **容错与鲁棒性增强**
   - 答案提取多策略回退：确保在模型输出格式不规范时仍能提取到有效答案
   - 训练数据格式验证：自动检测并修复 ChatML 格式问题
   - Pipeline 异常处理：求解器失败时优雅降级

4. **文档与方案完善**
   - 更新 `README.md`：反映 CoT 路线的架构变更和最新进度
   - 完善 `SCoRE2026_项目执行方案.md`：补充 CoT 方案细节和风险应对
   - 编写 `docs/plans/2026-07-03-scored-cot-reasoning.md`：详细的 CoT 实施计划

### 产出文件
| 文件 | 说明 |
|------|------|
| `scripts/generate_fusion_data.py` | 融合域数据增强 |
| `outputs/combined_train.json` | 3800 条合并训练集 |
| `docs/plans/2026-07-03-scored-cot-reasoning.md` | CoT 实施计划文档 |
| 更新 `README.md` | 项目进度更新 |

---

## Day 6（7月8日）—— 模型优化整合、撰写报告

### 完成内容

1. **端到端系统集成验证**
   - 验证 CoT Pipeline 三阶段数据流一致性：
     - `generate_cot_data.py` 输出 `{"id", "domain", "language", "text": "<chatml>"}` → `train_cot_model.py` 消费 `"text"` 字段 ✅
     - `train_cot_model.py` 产出 `checkpoints/cot_model/final/` → `run_cot_inference.py` 消费 `--model_path` ✅
     - 答案格式统一为 `{"answers": ["A", "B"]}`，与官方提交格式一致 ✅
   - 验证 ChatML 格式与 Qwen2.5 tokenizer 兼容性
   - 确认 LoRA adapter 正确加载到 base model

2. **消融实验设计**
   - 零样本基线：Qwen2.5-7B 不微调直接推理
   - Few-shot CoT：提供 3-5 个示例的 CoT prompt
   - LoRA 微调：CoT 训练数据微调后的模型
   - 模板解析器基线：Neuro-Symbolic 路线（已废弃）的性能参考
   - 对比维度：总体准确率、各领域准确率、融合域泛化能力

3. **提交文件准备**
   - 确认提交格式：`[{"id": "SCoRE2026-test-N", "answers": ["A", "B"]}]`
   - 编写格式校验脚本，确保输出符合在线评测系统要求
   - 确认不包含测试集信息泄露

4. **技术报告撰写**
   - 撰写项目技术报告框架（4 页，CCL 格式）
   - 内容涵盖：任务分析、技术方案演进（Neuro-Symbolic → CoT）、系统架构、实验结果与分析、讨论与展望

### 产出文件
| 文件 | 说明 |
|------|------|
| 更新 `main.py` | Pipeline 集成验证通过 |
| `progress.md` | 本进度报告 |

### 当前系统架构

```
训练阶段:
  训练集 (3600题) → [DeepSeek-V3 CoT推理] → 过滤(仅保留正确推理) → [LoRA微调 Qwen2.5-7B]

推理阶段:
  测试集 (1000题) → [微调后 Qwen2.5-7B] → CoT推理链 + 答案JSON

输出格式:
  ## Reasoning
  [逐步推理过程：实体识别 → 约束分析 → 逐项验证]
  ## Answer
  {"answers": ["A", "B"]}
```

---

## Day 7（7月9日）—— 准备答辩

### 完成内容

1. **答辩 PPT 准备**
   - 项目背景与任务介绍：SCoRE2026 评测任务、五大推理类型、核心挑战
   - 技术方案演进：Neuro-Symbolic → CoT 推理的决策过程与原因分析
   - 系统架构展示：三阶段 Pipeline（数据生成 → LoRA 微调 → 推理提交）
   - 实验结果与分析：各领域准确率对比、消融实验结论
   - 创新点总结：CoT 知识蒸馏、中英双语推理、多策略答案提取
   - 不足与展望：训练数据量限制、融合域泛化改进方向

2. **代码与文档整理**
   - 清理临时文件和调试代码
   - 统一代码注释风格，确保可复现性
   - 整理 `outputs/` 目录：保留最终提交文件和关键中间产物
   - 更新 `README.md` 至最终版本，包含完整的使用说明

3. **答辩演练**
   - 准备 5 分钟技术方案陈述
   - 准备常见问题应答：
     - 为什么放弃 Neuro-Symbolic 路线？（约束提取正确率仅 3.6%）
     - 如何处理训练-测试域偏移？（CoT 推理链天然跨域泛化）
     - 模型规模限制如何应对？（Qwen2.5-7B 刚好满足 8B 上限）
     - 中英双语如何处理？（统一 prompt 模板，不区分语言）
   - 准备 Demo 演示：从数据到推理的完整流程展示

4. **最终交付物汇总**

| 类别 | 交付物 | 状态 |
|------|--------|------|
| 代码 | 约束 Schema + 5 领域求解器 | ✅ |
| 代码 | 模板解析器 + 基线测试 | ✅ |
| 代码 | CoT 数据生成 + 模型微调 + 推理脚本 | ✅ |
| 代码 | 端到端 Pipeline + 答案验证器 | ✅ |
| 代码 | 融合域数据增强 | ✅ |
| 数据 | 测试集提交 JSON | ✅ |
| 文档 | 项目执行方案 | ✅ |
| 文档 | CoT 实施计划 | ✅ |
| 文档 | 进度报告（本文件） | ✅ |
| 文档 | 技术报告（4 页 CCL 格式） | 🔜 待提交 |
| 答辩 | PPT 演示文稿 | 🔜 待提交 |

---

## 项目总结

### 技术路线演进

```
Neuro-Symbolic (Day 1-3)           CoT Reasoning (Day 3-7)
┌─────────────────────┐            ┌──────────────────────────┐
│ NL → 约束JSON → 求解 │  ──放弃──→ │ NL → CoT推理链 → 答案JSON │
│  (正确率 3.6%)       │            │ (端到端，避免中间误差)     │
└─────────────────────┘            └──────────────────────────┘
```

### 关键决策与经验

1. **快速试错、果断转向**：Neuro-Symbolic 路线在约束提取阶段即暴露致命缺陷（3.6% 正确率），Day 3 即决定转向 CoT 路线，避免了在错误方向上持续投入
2. **保留中间产物**：Neuro-Symbolic 的符号求解器和模板解析器保留作为消融实验基线，为方案对比提供了有力支撑
3. **CoT 知识蒸馏**：用强模型（DeepSeek-V3）的推理能力训练小模型（Qwen2.5-7B），在满足 8B 参数限制的前提下最大化推理性能
4. **质量优先的训练数据**：仅保留预测正确的推理链作为训练数据，确保模型学习的是正确的推理模式而非错误的猜测

### 后续改进方向

- 增加 CoT 训练数据量（当前使用 API 预算限制，可扩展到更多样本）
- 尝试更强的 Teacher 模型（如 DeepSeek-R1）生成更高质量的推理链
- 探索 CoT + 符号求解器混合方案（两者互补）
- 针对融合域做专项数据增强和课程学习

---

> 📅 报告日期：2026年7月3日 - 7月9日
> 📝 最后更新：2026年7月3日
