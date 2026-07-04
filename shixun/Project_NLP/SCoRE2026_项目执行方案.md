# SCoRE2026 项目执行方案

> **任务：** CCL 2026 任务十 —— 基于情景的常识推理评测（SCoRE）
> **主办方：** 北京大学 + 华为
> **难度等级：** B级（中高难度，20/30分）

---

## 一、任务概述

SCoRE2026 旨在评估大语言模型在常识场景下的复杂逻辑推理能力。核心挑战是将自然语言描述的情景转化为结构化约束，进行多步逻辑推理，最终选择正确答案。

### 1.1 五类推理任务

| 类型 | 核心能力 | 训练集数量 | 测试集（单域） | 测试集（融合域） |
|------|---------|-----------|--------------|----------------|
| **时间推理** | 事件时间线重建、相对时间偏移计算 | 1000 | 100 | time+social: 200, time+nature: 200 |
| **空间推理** | 多物体空间布局推理、方位关系推导 | 1000 | 100 | space+social: 200, space+nature: 75 |
| **自然推理** | 物体属性分类、排除法匹配 | 1000 | 100 | time+nature: 200, space+nature: 75 |
| **社会推理** | 亲属/职场关系网络推导 | 500 | 25 | time+social: 200, space+social: 200 |
| **融合推理** | 跨域联合约束求解 | 100 | -- | 共675 |

### 1.2 关键约束条件

| 约束项 | 要求 |
|--------|------|
| **模型规模** | Dense模型 ≤ 8B；MoE模型总参 ≤ 30B，每token激活 ≤ 3B |
| **外部数据** | 禁止使用SCoRE2026以外的任何数据集 |
| **测试数据** | 禁止用于训练、微调或作为prompt示例 |
| **评估指标** | Accuracy = 正确数 / 总题数 |
| **提交格式** | JSON文件，通过在线评测系统提交 |

---

## 二、核心挑战分析

### 🔴 挑战1：训练-测试域偏移（最关键）

```
训练集分布：               测试集分布：
┌──────────────┐          ┌────────────────────┐
│ 单域 97.2%   │    →     │ 融合域 67.5%        │
│ 融合域 2.8%  │          │ 单域 32.5%          │
└──────────────┘          └────────────────────┘
```

**训练集几乎全是单域问题，测试集以融合域为主。** 这是本项目最大的技术挑战——模型必须在单域数据上训练出可迁移的推理能力，在测试时泛化到多域交叉场景。

### 🟠 挑战2：多步约束求解

每个问题本质上是一个**约束满足问题（CSP）**：
- 需从自然语言中提取约束条件
- 构建并求解关系图/时间线/空间布局
- 将求解结果映射回选项

### 🟡 挑战3：双语推理

中英文各约一半，模型需同时掌握两种语言的推理模式。中文涉及更复杂的亲属称谓（如"姐夫"、"岳母"、"儿媳"等），英文涉及不同的时空表达习惯。

### 🟢 挑战4：多答案问题

训练集中23.6%（848/3600）的题目有多个正确答案（不定项选择），需要模型判断哪些选项同时满足条件。

---

## 三、技术方案设计

### 3.1 总体架构：Neuro-Symbolic 混合系统

```
输入文本（自然语言）
    │
    ▼
┌─────────────────────────────┐
│ 模块1: 约束解析器 (LLM)      │  ← 微调后的LLM
│  NL → 结构化约束             │
│  输出：JSON约束列表          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 模块2: 符号求解器 (Python)   │  ← 确定性算法
│  约束传播 + 回溯搜索         │
│  输出：完整赋值方案          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 模块3: 答案验证器 (Python)   │  ← 规则引擎
│  将赋值方案与选项逐一比对     │
│  输出：正确选项列表          │
└─────────────────────────────┘
```

### 3.2 为什么选择 Neuro-Symbolic 方案

| 纯LLM方案 | Neuro-Symbolic方案 |
|-----------|-------------------|
| 在复杂多步推理中容易出错 | 符号求解器保证100%正确（如有正确约束） |
| 对融合域泛化能力不足 | 约束求解天然跨域通用 |
| 需要大量融合域训练数据 | 只需LLM做好约束解析，无需融合域训练数据 |
| 黑盒，难以调试 | 求解过程完全可解释、可验证 |

**核心理念：** LLM负责"理解"（从NL到结构化约束），符号求解器负责"推理"（约束满足）。将LLM不擅长的精确多步推理交给确定性算法。

### 3.3 候选模型选择

| 模型 | 参数 | 优势 | 劣势 |
|------|------|------|------|
| **Qwen2.5-7B-Instruct** | 7B | 中英双语强、指令跟随好、开源生态好 | 相对较新 |
| **DeepSeek-R1-Distill-Qwen-7B** | 7B | 推理能力强、CoT原生支持 | 推理速度较慢 |
| **Mistral-7B-Instruct** | 7B | 英文强 | 中文弱 |
| **Llama-3.1-8B-Instruct** | 8B | 综合能力强 | 中文相对弱 |

**推荐：Qwen2.5-7B-Instruct** 作为主模型，DeepSeek-R1-Distill-Qwen-7B 作为对比实验。

---

## 四、分阶段实施计划

### Phase 0：环境搭建与数据分析（第1-2天）

**目标：** 完成开发环境搭建，深入理解数据模式

```
任务清单：
├── 0.1 GPU环境配置（CUDA, PyTorch, vLLM/Transformers）
├── 0.2 数据深度分析脚本
│   ├── 各领域问题模式归类
│   ├── 约束类型统计（相对时间、空间方位、亲属关系...）
│   ├── 中英文差异分析
│   └── 错误类型预判
└── 0.3 基线测试
    ├── Qwen2.5-7B 零样本准确率
    ├── Qwen2.5-7B Few-shot CoT准确率
    └── DeepSeek-R1-7B 零样本准确率
```

**数据预处理要点：**

```python
# 数据字段标准化
{
    "id": "SCoRE2026-train-1",
    "domain": "time",           # 单域或融合域
    "language": "en",           # cn / en
    "text": "...",              # 情景描述
    "question": "...",          # 填空式/选择式问题
    "options": {"A": ..., ...}, # 4选项
    "answers": ["A", "B"],      # 可为多选
    # 以下为解析后添加的字段
    "entities": [...],          # 抽取的实体列表
    "constraints": [...],       # 结构化约束
    "question_type": "fill_blank" | "select_correct" | "select_incorrect"
}
```

---

### Phase 1：约束解析器训练（第3-7天）

**目标：** 训练LLM将自然语言情景精确解析为结构化约束

#### 1.1 约束模式定义

为每种领域定义标准化的约束JSON Schema：

**时间约束：**
```json
{
  "domain": "time",
  "entities": ["event_1", "event_2", "星期三", "星期四"],
  "constraints": [
    {"type": "absolute", "event": "打羽毛球", "time": "星期三"},
    {"type": "relative", "event1": "阅读科幻小说", "event2": "开组会", 
     "relation": "after", "offset": 1, "unit": "day"}
  ]
}
```

**空间约束：**
```json
{
  "domain": "space",
  "entities": ["茶花", "水仙", "波斯菊", "月季", "君子兰", "郁金香"],
  "structure": {"type": "grid", "rows": 3, "cols": 2, "orientations": ["东", "西"]},
  "constraints": [
    {"type": "above", "entity1": "君子兰", "entity2": "郁金香", "gap": 1},
    {"type": "adjacent", "entity1": "郁金香", "entity2": "月季", "direction": "left"},
    {"type": "position", "entity": "茶花", "row": 2, "col": "东"}
  ]
}
```

**社会约束：**
```json
{
  "domain": "social",
  "entities": ["赵芳", "赵丹", "赵秀兰", "赵威", ...],
  "constraints": [
    {"type": "kinship", "person1": "赵芳", "person2": "赵丹", 
     "relation": "姐夫的儿媳"}
  ]
}
```

**自然约束：**
```json
{
  "domain": "nature",
  "entities": ["莴苣", "南瓜", "胡萝卜", "花生"],
  "positions": ["1号田", "2号田", "3号田", "4号田"],
  "attributes": {
    "莴苣": {"类别": "蔬菜", "花色": "?", "可食用部分": "?"},
    "南瓜": {"类别": "蔬菜", "花色": "黄色", "可食用部分": "?"}
  },
  "constraints": [
    {"type": "property_match", "entity": "1号田作物", "property": "花色", "value": "黄色"},
    {"type": "category", "entity": "4号田作物", "category": "蔬菜"}
  ]
}
```

#### 1.2 约束解析训练数据构建

由于官方训练集只提供了QA对，没有中间约束标注，需要：

**方案A（推荐）：** 人工标注 200-300 条约束 + LLM辅助扩展
1. 每个领域人工标注 50 条样本的结构化约束
2. 用这些标注样本few-shot prompt GPT-4/Claude生成剩余样本的约束标注
3. 人工抽检200条确保质量
4. 得到3600条带约束标注的训练数据

**方案B（快速）：** 全自动约束生成
1. 设计每个领域的约束解析模板
2. 用规则+正则提取部分约束
3. 用强LLM（如DeepSeek-V3）批量生成约束标注
4. 用符号求解器验证标注质量（能正确求解=高质量标注）

#### 1.3 微调训练

```yaml
模型: Qwen2.5-7B-Instruct
训练框架: LLaMA-Factory / Unsloth
训练策略:
  - LoRA微调 (r=16, alpha=32)
  - 学习率: 2e-4
  - Batch size: 16
  - Epochs: 3
  - 输入: 自然语言情景文本
  - 输出: JSON格式的结构化约束
训练数据: 3600条 + 数据增强
验证集: 从训练集中按领域分层抽取10%
```

---

### Phase 2：符号求解器开发（第4-10天）

**目标：** 为每个领域开发专用的约束求解算法

#### 2.1 各领域求解策略

**时间推理求解器：**
```
算法：约束传播 + 拓扑排序
1. 解析绝对时间约束（如"星期三，他打羽毛球"）
2. 解析相对时间约束（如"开组会之后1天，阅读科幻小说"）
3. 构建时间有向图，边权重为时间偏移
4. 选择一个锚点事件，DFS/拓扑排序推导所有事件时间
5. 处理周期性问题（每周7天循环）
6. 如有多个解，保留所有可能解
```

**空间推理求解器：**
```
算法：约束满足问题(CSP) + 回溯搜索
1. 构建空间结构（网格/圆形/线性）
2. 列出所有实体和位置
3. 逐条应用约束，缩小可能位置
4. 回溯搜索找到所有满足约束的布局
5. 验证唯一性（多解时需要额外推理）
```

**社会推理求解器：**
```
算法：关系图构建 + 传递闭包 + 性别推理
1. 解析亲属关系为有向图（带关系标签）
2. 计算关系传递闭包
3. 从已知关系推导隐含关系
4. 从称谓推断性别（如"岳母"→女性→丈母娘）
5. 验证一致性和唯一性
```

**自然推理求解器：**
```
算法：属性矩阵 + 约束传播 + 排除法
1. 构建实体-属性矩阵
2. 逐条应用约束，标记已知属性
3. 利用排除法缩小可能匹配
4. 当某个实体只有一种可能时，锁定并传播
5. 回溯搜索解决歧义
```

**融合推理求解器：**
```
算法：分阶段求解 + 信息传递
1. 识别涉及的领域（如time+social）
2. 先求解一个领域（如social→确定人物关系）
3. 将求解结果作为已知信息注入第二个领域
4. 再求解第二个领域（如time→确定时间安排）
5. 综合两个领域的结果
```

#### 2.2 求解器验证

用训练集的3600条数据验证求解器：约束正确时，求解器应100%得出正确答案。若不能，则说明约束标注有问题或求解器有bug。

---

### Phase 3：融合域数据增强（第7-11天）

**目标：** 解决训练集中融合域数据不足的问题

#### 3.1 程序化数据生成

利用已知的单一领域模板，自动组合生成融合域训练数据：

```python
# 伪代码：time+social融合数据生成
def generate_time_social_fusion():
    # 1. 随机生成一个社交关系网络（5-8人）
    social_graph = generate_random_kinship_network(num_people=6)
    
    # 2. 随机生成时间约束（用社交关系中的称谓指代人）
    people_refs = derive_references_from_social_graph(social_graph)
    # e.g., "Maria Miller's leader", "Kevin Taylor's teacher"
    
    temporal_constraints = generate_temporal_constraints(refs=people_refs)
    
    # 3. 组合为社会+时间融合问题
    # 先描述社交关系，再描述时间安排
    text = format_social_description(social_graph) + format_temporal_description(temporal_constraints)
    
    # 4. 生成问题和选项
    question, options, answers = generate_qa(social_graph, temporal_constraints)
    
    return {"text": text, "question": question, "options": options, "answers": answers, "domain": "time+social"}
```

#### 3.2 数据增强策略

| 增强类型 | 方法 | 目标数量 |
|---------|------|---------|
| 程序生成融合数据 | 模板组合 + 随机采样 | 1500条 |
| 反向翻译增强 | 中→英→中 / 英→中→英 | 1000条 |
| 实体替换增强 | 替换人名/物名/时间表达 | 500条 |
| 约束扰动增强 | 微调约束条件生成新题 | 500条 |

---

### Phase 4：端到端系统集成与优化（第10-14天）

**目标：** 将约束解析器和符号求解器集成为端到端系统

#### 4.1 系统Pipeline

```python
class SCOREPipeline:
    def __init__(self, model, solvers):
        self.constraint_parser = ConstraintParser(model)  # LLM
        self.solvers = {
            "time": TemporalSolver(),
            "space": SpatialSolver(),
            "social": SocialSolver(),
            "nature": NatureSolver(),
        }
        self.answer_verifier = AnswerVerifier()
    
    def solve(self, text, question, options, domain):
        # Step 1: LLM解析约束
        constraints = self.constraint_parser.parse(text, domain)
        
        # Step 2: 根据域类型调用求解器
        if "+" in domain:
            # 融合域：分阶段求解
            domain1, domain2 = domain.split("+")
            result1 = self.solvers[domain1].solve(constraints[domain1])
            # 将结果1注入求解器2的上下文
            result2 = self.solvers[domain2].solve(
                constraints[domain2], context=result1
            )
            solution = {**result1, **result2}
        else:
            solution = self.solvers[domain].solve(constraints)
        
        # Step 3: 验证每个选项
        correct_options = self.answer_verifier.verify(
            question, options, solution
        )
        
        return correct_options
```

#### 4.2 容错机制

LLM约束解析可能出现错误，需要多层次的容错机制：

```
容错策略（按优先级）：
├── L1: 约束验证 —— 检查JSON格式、实体一致性、约束合法性
│   └── 不合法时：自动修正或请求LLM重新生成
├── L2: 求解失败回退 —— 符号求解器无法找到解
│   └── 回退到纯LLM CoT推理
├── L3: 多解歧义 —— 符号求解器找到多个可行解
│   └── 调用LLM判断最合理的解
└── L4: 答案置信度评估 —— 对比符号求解器与LLM直接推理结果
    └── 不一致时：以符号求解器为准（通常更可靠）
```

#### 4.3 备选方案：纯LLM微调方案

如果Neuro-Symbolic方案的约束解析准确率不足，作为备选：

```yaml
方案: 全量微调LLM + CoT推理
模型: Qwen2.5-7B-Instruct
训练数据:
  - 原始3600条训练数据
  - 程序生成的1500条融合域数据
  - CoT推理链作为训练目标
训练方式:
  - 输入: [text] + [question] + [options]
  - 输出: [step-by-step reasoning] + [answer: A, B]
评估: 在验证集上对比Neuro-Symbolic方案
```

---

### Phase 5：测试与提交（第14-17天）

#### 5.1 本地验证策略

由于测试集无答案，使用以下方法评估泛化能力：

1. **训练集K-fold验证**：5-fold交叉验证，特别关注融合域fold
2. **留出融合域验证集**：从训练集的100条space+nature数据中留出30条
3. **程序生成的伪测试集**：用数据增强流程生成与测试集同分布的伪测试集

#### 5.2 提交文件格式

```json
[
  {
    "id": "SCoRE2026-test-1",
    "answers": ["A", "C"]
  },
  {
    "id": "SCoRE2026-test-2",
    "answers": ["B"]
  }
]
```

#### 5.3 消融实验计划

| 实验 | 配置 | 目的 |
|------|------|------|
| 基线 | Qwen2.5-7B 零样本 | 确定难度下限 |
| CoT | Qwen2.5-7B + 少样本CoT | 评估推理链效果 |
| 微调 | Qwen2.5-7B LoRA微调（无约束解析） | 评估微调收益 |
| N-S基础 | Neuro-Symbolic（规则解析约束） | 评估符号求解器贡献 |
| N-S完整 | Neuro-Symbolic（LLM解析约束 + 符号求解） | 评估完整方案 |
| 融合数据 | N-S完整 + 融合域增强数据 | 评估数据增强收益 |

---

## 五、技术要点详解

### 5.1 中文亲属关系处理

中文社会推理涉及复杂的亲属称谓网络，需要专门处理：

```python
# 亲属关系传递闭包推理规则
KINSHIP_RULES = {
    # 基本关系传递
    ("A的爸爸", "B的爸爸"): lambda A, B: f"{A}是{B}的爷爷/外公",
    ("A的姐姐", "B的妈妈"): lambda A, B: f"{A}是{B}的姨",
    
    # 姻亲关系
    ("A的老公", "B的爸爸"): lambda A, B: f"{A}是{B}的妈妈",
    ("A的儿媳", "B的妈妈"): lambda A, B: f"{A}是{B}的嫂子/弟媳",
    
    # 隔代关系
    ("A的奶奶", "B的妈妈"): lambda A, B: f"{A}是{B}的曾祖母",
}
```

### 5.2 时间周期处理

时间是循环的（每周7天），需要模运算：

```python
def resolve_temporal_chain(events, constraints):
    """
    将相对时间约束转化为绝对时间线
    处理每周循环：使用模7运算
    """
    graph = defaultdict(list)
    for c in constraints:
        # "A after B by N days" → edge: B→A with weight N
        graph[c["event1"]].append((c["event2"], c["offset"]))

    # 拓扑排序 + 约束传播
    timeline = {}
    anchor = find_anchor_event(events, constraints)  # 有绝对时间的
    timeline[anchor] = absolute_time[anchor]

    # BFS/DFS传播
    queue = [anchor]
    while queue:
        curr = queue.pop(0)
        for next_event, offset in graph[curr]:
            if next_event not in timeline:
                timeline[next_event] = (timeline[curr] + offset) % 7
                queue.append(next_event)

    return timeline
```

### 5.3 空间布局编码

```python
class SpatialGrid:
    """3层×2列的空间网格，用于花架/货架类问题"""
    def __init__(self, rows=3, cols=2):
        self.grid = [[None for _ in range(cols)] for _ in range(rows)]

    def apply_constraint(self, constraint):
        ctype = constraint["type"]
        if ctype == "above":
            # entity1在entity2正上方，可能隔N层
            self.constrain_above(constraint["entity1"],
                                 constraint["entity2"],
                                 constraint.get("gap", 0))
        elif ctype == "adjacent":
            # entity1是entity2的左邻/右邻
            self.constrain_adjacent(constraint["entity1"],
                                    constraint["entity2"],
                                    constraint.get("direction", "left"))
        # ... 更多约束类型

    def solve(self):
        """回溯搜索找到所有满足约束的布局"""
        entities = self.get_all_entities()
        positions = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        return self._backtrack(entities, positions, {})
```

---

## 六、项目时间线

```
Week 1 (Day 1-7):
  Day 1-2: 环境搭建 + 基线测试 + 数据分析
  Day 3-5: 约束Schema设计 + 人工标注(200条)
  Day 5-7: LLM辅助扩展约束标注(3600条全量)

Week 2 (Day 4-10):
  Day 4-6: 时间推理求解器 + 空间推理求解器
  Day 7-8: 社会推理求解器 + 自然推理求解器
  Day 9-10: 融合推理求解器 + 求解器全量验证

Week 3 (Day 7-14):
  Day 7-9: 融合域数据自动生成
  Day 10-12: LoRA微调约束解析器
  Day 12-14: 端到端系统集成 + 容错机制

Week 4 (Day 14-17):
  Day 14-15: 消融实验 + 本地验证
  Day 15-16: 系统调优 + 最终测试
  Day 17: 提交 + 技术报告初稿
```

### 甘特图

```
         W1        W2        W3        W4
         1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7
环境搭建 ██
数据分析 ██
基线测试   ██
约束标注     ████████
时间求解器     ██████
空间求解器         ████
社会求解器           ████
自然求解器           ████
融合求解器             ████
数据增强               ██████
LoRA微调                  ██████
系统集成                    ██████
消融实验                      ████
调优提交                        ████
```

---

## 七、风险与对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| LLM约束解析准确率不足 | 中 | 高 | 增加标注数据量；回退纯LLM微调方案 |
| 融合域泛化效果差 | 高 | 高 | 程序化生成大量融合数据；多阶段训练 |
| 中文亲属关系过于复杂 | 中 | 中 | 构建中文亲属关系知识库；专项数据增强 |
| 8B模型容量不够 | 低 | 中 | 选择MoE架构模型；模型蒸馏 |
| 符号求解器覆盖不全 | 中 | 中 | 求解失败时回退LLM推理；持续迭代 |
| 时间不足 | 中 | 高 | 优先保证核心Pipeline；简化非关键模块 |

---

## 八、交付物清单

- [ ] 约束解析器（LoRA微调权重 + 推理代码）
- [ ] 各领域符号求解器（Python模块）
- [ ] 端到端推理Pipeline
- [ ] 数据增强脚本
- [ ] 消融实验报告
- [ ] 测试集提交JSON
- [ ] 技术报告（4页，CCL格式）
- [ ] 可复现的Docker镜像或requirements.txt

---

## 九、参考资源

| 资源 | 链接 |
|------|------|
| 任务官网 | https://pku-space.github.io/SCoRE2026/ |
| 评测主席 | 詹卫东、穗志方（北京大学） |
| 联系邮箱 | hunan@stu.pku.edu.cn |
| Qwen2.5 | https://github.com/QwenLM/Qwen2.5 |
| LLaMA-Factory | https://github.com/hiyouga/LLaMA-Factory |
| Unsloth | https://github.com/unslothai/unsloth |

---

> 📅 创建日期：2026年7月3日
> 📝 方案版本：v1.0
