# 智能职位推荐系统 - Design Spec

> Human-readable design narrative. Machine-readable execution contract: `spec_lock.md`.

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | 智能职位推荐系统_课程项目汇报 |
| **Canvas Format** | PPT 16:9 (1280×720) |
| **Page Count** | 19 |
| **Design Style** | soft-rounded — 圆润卡片、现代亲和、专业简洁 |
| **Target Audience** | 课程答辩评委（教师 + 同学） |
| **Use Case** | 社交网络数据分析与处理课程期末项目汇报 |
| **Delivery Purpose** | `presentation` — 教室投影演示，大字号、少文字、多视觉 |
| **Content Strategy** | balanced — 保留原PPT核心叙事结构和文字内容，为8张系统截图合理分配展示空间；第4页团队分工页原封不动保留 |
| **Created Date** | 2026-06-25 |

---

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280×720 |
| **viewBox** | `0 0 1280 720` |
| **Margins** | left/right 60px, top/bottom 50px |
| **Content Area** | 1160×620 |

---

## III. Visual Theme

### Theme Style

- **Mode**: `showcase` — 功能演示为主，每页一个核心视觉元素（截图或关键数据），最小化文案
- **Visual style**: `soft-rounded` — 圆角卡片（rx 12-16）、柔和阴影、亲和现代
- **Theme**: Light theme — 白色背景 + 翠绿色调
- **Tone**: 专业、学术、技术创新 — 面向课程答辩的正式但不沉闷的基调

### Color Scheme

继承原PPTX的翠绿色调，与华中农业大学视觉风格契合：

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#FFFFFF` | 页面背景 |
| **Secondary bg** | `#E8F5E9` | 卡片背景、区块底色 |
| **Primary** | `#2E7D32` | 标题装饰、关键图标、章节页主色 |
| **Accent** | `#1B5E20` | 数据高亮、链接、重点标记 |
| **Secondary accent** | `#81C784` | 次要强调、渐变过渡 |
| **Body text** | `#333333` | 正文 |
| **Secondary text** | `#666666` | 说明文字、标注 |
| **Tertiary text** | `#999999` | 补充信息、页脚 |
| **Border/divider** | `#C8E6C9` | 卡片边框、分割线 |
| **Success** | `#4CAF50` | 正面指标 |
| **Warning** | `#FF9800` | 警示标记 |

### Gradient Scheme

```xml
<linearGradient id="titleGradient" x1="0%" y1="0%" x2="100%" y2="0%">
  <stop offset="0%" stop-color="#2E7D32"/>
  <stop offset="100%" stop-color="#81C784"/>
</linearGradient>

<radialGradient id="bgDecor" cx="85%" cy="15%" r="45%">
  <stop offset="0%" stop-color="#2E7D32" stop-opacity="0.08"/>
  <stop offset="100%" stop-color="#2E7D32" stop-opacity="0"/>
</radialGradient>
```

---

## IV. Typography System

### Font Plan

**Typography direction**: 现代CJK无衬线 — 统一使用 Microsoft YaHei，简洁专业。

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | `"Microsoft YaHei", "PingFang SC"` | `Arial` | `sans-serif` |
| **Body** | `"Microsoft YaHei", "PingFang SC"` | `Arial` | `sans-serif` |
| **Emphasis** | `"Microsoft YaHei"` | `Arial` | `sans-serif` |
| **Code** | — | `Consolas, "Courier New"` | `monospace` |

**Per-role font stacks**:
- Title: `"Microsoft YaHei", "PingFang SC", Arial, sans-serif`
- Body: `"Microsoft YaHei", "PingFang SC", Arial, sans-serif`
- Emphasis: same as Body
- Code: `Consolas, "Courier New", monospace`

### Font Size Hierarchy

**Baseline (unitless px)**: Body = 32px (`presentation` mode)

| Role | Ratio | Locked px | Weight |
| ---- | ----- | --------- | ------ |
| Cover title | 2.25x | 72 | Bold |
| Chapter opener | 1.75x | 56 | Bold |
| Page title | 1.5x | 48 | Bold |
| Subtitle | 1.2x | 38 | SemiBold |
| Lead / core message | 1.0x | 32 | Medium |
| **Body** | **1x** | **32** | Regular |
| Annotation / caption | 0.7x | 22 | Regular |
| Footnote / page number | 0.5x | 16 | Regular |

---

## V. Layout Principles

### Page Structure

- **Header area**: Title bar, top 100px, left-aligned title + right-aligned page number
- **Content area**: 620px height from y=100 to y=720, flexible layout
- **Footer area**: Bottom 40px, page number + project name

### Layout Pattern Library

| Pattern | Suitable Scenarios |
| ------- | ----------------- |
| **Single column centered** | Cover, chapter dividers, closing |
| **Asymmetric split (screenshot + text)** | Feature demo pages — screenshot on right (785x600 area), text on left |
| **Top-bottom split** | Dual-screenshot pages (login+dashboard, upload+analysis) |
| **Three-column cards** | Tech stack, feature lists |
| **Z-pattern / waterfall** | Architecture flow, algorithm pipeline |
| **Full-bleed + floating text** | Chapter dividers with green background |

### Spacing

| Element | Value |
| ------- | ----- |
| Safe margin | 60px |
| Content block gap | 32px |
| Card gap | 24px |
| Card padding | 28px |
| Card border radius | 14px |
| Icon-text gap | 12px |

---

## VI. Icon Usage Specification

### Source

- **Library**: `tabler-outline` — 线条图标，轻盈现代，适合屏幕投影
- **Stroke width**: 2px deck-wide

### Icon Inventory

| Purpose | Icon Path | Pages |
| ------- | --------- | ----- |
| 主页/仪表盘 | `tabler-outline/home` | P02, P09 |
| 搜索/查询 | `tabler-outline/search` | P02, P10 |
| 团队 | `tabler-outline/users` | P02, P04 |
| 数据库 | `tabler-outline/database` | P05, P07 |
| AI/智能 | `tabler-outline/brain` | P02, P17 |
| 图表/数据 | `tabler-outline/chart-bar` | P02, P06 |
| 时间线 | `tabler-outline/timeline` | P02, P12 |
| 对话/问答 | `tabler-outline/message-circle` | P02, P13 |
| 文档/简历 | `tabler-outline/file-text` | P02, P14 |
| 上传 | `tabler-outline/upload` | P14 |
| 亮点/特色 | `tabler-outline/star` | P19 |
| 确认/完成 | `tabler-outline/check` | P19 |
| 箭头/流程 | `tabler-outline/arrow-right` | P16 |
| 代码/技术 | `tabler-outline/code` | P06 |
| 服务器 | `tabler-outline/server` | P05 |
| 图谱/关系 | `tabler-outline/graph` | P07, P10 |
| 网络 | `tabler-outline/network` | P05 |
| 分支/版本 | `tabler-outline/git-branch` | P06 |
| 建筑/企业 | `tabler-outline/building` | P03 |
| 分层架构 | `tabler-outline/layers-linked` | P05 |

---

## VII. Visualization Reference List

Catalog read: 71 templates

This deck is screenshot-demonstration heavy and data-light — no traditional data charts (bar/line/pie) are needed. Structural templates are used for:

| Page | Template | Path | Summary-quote (verbatim) | Usage |
| ---- | -------- | ---- | ------------------------ | ----- |
| P05 | vertical_pillars | `templates/charts/vertical_pillars.svg` | "Pick for layered architecture, stack diagrams, or capability models with 3-5 tiers." | Four-tier system architecture (Frontend → AI → Backend → Data) |
| P16 | chevron_chain_with_tail | `templates/charts/chevron_chain_with_tail.svg` | "Pick for linear 4-6 step value chains or sequential processes." | Five-step recommendation algorithm pipeline |

**Runners-up considered** (fewer than 3 viz pages — structural pages only):
- `process_flow` | rejected for P16: chevron_chain is more specific for labeled sequential steps with side annotations
- `hub_inward_arrows` | rejected for P05: pillars better capture layered stack than hub-and-spoke
- `numbered_steps` | rejected for P16: too generic; algorithm flow needs direction + weight annotations

---

## VIII. Image Resource List

All 8 screenshots are user-provided (Existing), acquired from the running application. All are tagged `no-crop` to preserve pixel-perfect UI detail.

| Filename | Dimensions | Ratio | Purpose | Type | Layout pattern | Acquire Via | Status | Reference |
| -------- | --------- | ----- | ------- | ---- | -------------- | ----------- | ------ | --------- |
| 00-login.png | 1440×1100 | 1.31 | 系统登录入口 — 展示前端正常加载与用户认证入口 | Screenshot | #2 left-third image with right text + #21 rounded rectangle crop | user | Existing | 登录页截图，展示系统入口和用户认证界面 |
| 01-dashboard.png | 1440×1100 | 1.31 | 首页仪表盘 — 展示左侧导航、顶部工具栏和主框架布局 | Screenshot | #2 left-third image with right text + #21 rounded rectangle crop | user | Existing | 登录后主界面，展示系统导航结构和整体布局 |
| 02-knowledge-graph.png | 1440×1100 | 1.31 | 知识图谱可视化 — 岗位-技能-学历-经验-公司-城市关系网络 | Screenshot | #2 left-third image with right text + #21 rounded rectangle crop | user | Existing | ECharts力导向图，节点按实体类型着色，展示图结构关系 |
| 03-career-recommendation.png | 1440×1100 | 1.31 | 职业推荐系统 — 输入描述后返回推荐岗位表格+关系图谱 | Screenshot | #2 left-third image with right text + #21 rounded rectangle crop | user | Existing | 推荐结果表格与ECharts图谱同步展示 |
| 04-career-timeline.png | 1440×1100 | 1.31 | 职业时间线分析 — AI生成的职业四阶段发展路径 | Screenshot | #2 left-third image with right text + #21 rounded rectangle crop | user | Existing | Element Plus时间线组件展示职业发展阶段 |
| 05-ai-chat.png | 1440×1100 | 1.31 | AI智能问答 — 聊天界面，Markdown渲染，SSE流式响应 | Screenshot | #2 left-third image with right text + #21 rounded rectangle crop | user | Existing | 聊天界面展示AI问答交互效果 |
| 06-resume-upload.png | 1440×1100 | 1.31 | 简历上传入口 — 支持PDF/Word/图片格式上传 | Screenshot | #5 top-bottom band split + #21 rounded rectangle crop | user | Existing | 上传组件界面，展示多格式简历上传入口 |
| 07-resume-analysis.png | 1440×1100 | 1.31 | 简历分析结果 — AI生成的职业匹配卡片（职业名/概率/市场/行情） | Screenshot | #5 top-bottom band split + #21 rounded rectangle crop | user | Existing | 三列分析卡片展示AI职业匹配建议 |

---

## IX. Content Outline

### Part 1: 项目概述 (P01–P07)

#### P01 - 封面

- **Cover impact**: 课程名称 + 项目主题双标题，翠绿渐变底纹 + 右下角装饰性放射渐变，大字突出"职业关系图谱与智能推荐系统"
- **Layout**: 单列居中，标题在页面中上，副标题/机构信息在下方
- **Title**: 社交网络数据分析与处理 · 课程项目汇报
- **Subtitle**: 职业关系图谱与智能推荐系统
- **Info**: 华中农业大学 · 信息学院

#### P02 - 目录

- **Layout**: 四象限矩阵 — 左上Part 1 项目概述、右上Part 2 核心功能演示、左下Part 3 推荐算法与大模型、右下Part 4 总结
- **Title**: 目 录 CONTENTS
- **Core message**: 汇报围绕项目概述、功能演示、算法与模型、总结四大板块展开
- **Content**:
  - Part 1 项目概述与技术架构 — 系统定位 · 团队分工 · 架构 · 技术栈 · 数据图谱
  - Part 2 核心功能演示 — 知识图谱 · 职业推荐 · 时间线分析 · AI问答 · 简历分析
  - Part 3 推荐算法与大模型 — 推荐流程 · 匹配评分 · 大语言模型集成
  - Part 4 总结 — 项目亮点与成果

#### P03 - 课程项目定位

- **Layout**: 三列卡片 — 知识图谱构建 / 智能推荐引擎 / AI辅助分析，上方项目背景文字
- **Title**: 课程项目定位 PROJECT POSITIONING
- **Core message**: 面向就业场景的关系网络分析与职业推荐系统，用知识图谱+大模型帮助用户理解职业发展路径
- **Content**:
  - 项目背景与目标: 面向就业场景的关系网络分析与职业推荐系统。将岗位、公司、城市、技能、学历、经验等信息组织成知识图谱，结合检索、推荐和大语言模型问答，帮助用户理解职业要求和发展路径。
  - 知识图谱构建 — Neo4j图数据库存储实体关系 · 803条岗位数据 · 6类实体节点 · 5种关系类型
  - 智能推荐引擎 — 多维匹配评分与Top-K排序 · AI驱动推荐
  - AI辅助分析 — 大模型驱动问答与简历解析

#### P04 - 团队分工

- **Layout**: 原PPT第4页原封不动保留（用户明确要求）。两人并列卡片。
- **Title**: 团队分工 TEAM WORK DISTRIBUTION
- **Core message**: 两人协作完成项目查阅管理、开发实现与测试优化
- **Content**: [原PPT第4页全部内容原样保留]

#### P05 - 系统整体架构

- **Layout**: 四层垂直堆叠架构图，从上到下：前端展示层 → AI分析引擎 → 业务服务层 → 数据存储层
- **Title**: 系统整体架构 SYSTEM ARCHITECTURE
- **Core message**: 三服务协作架构 — Vue前端 + Spring Boot业务后端 + Flask AI引擎，底层MySQL+Neo4j+Redis
- **Visualization**: vertical_pillars
- **Content**:
  - 前端展示层: Vue 3 + Element Plus + ECharts 5 — 单页应用 · 动态路由 · 图谱可视化
  - AI 分析引擎: Flask + DeepSeek-V3 + Neo4j — 推荐引擎 · 简历解析 · AI问答 · 时间线分析
  - 业务服务层: Spring Boot 3.4 + MyBatis-Plus + JWT — 用户管理 · 职位CRUD · 鉴权拦截
  - 数据存储层: MySQL 8.0 + Neo4j + Redis — 业务数据 · 图数据库 · 会话缓存

#### P06 - 技术栈总览

- **Layout**: 三列卡片 — 前端技术 / 后端技术 / AI与数据
- **Title**: 技术栈总览 TECHNOLOGY STACK
- **Core message**: Vue 3 + Spring Boot + Flask 三端协作，覆盖前端、业务、AI全链路
- **Content**:
  - 前端技术: Vue 3 渐进式框架 · Element Plus UI组件库 · ECharts 5 数据可视化 · Axios HTTP客户端
  - 后端技术: Spring Boot 3.4 Java业务后端 · MyBatis-Plus 3.5 ORM框架 · JWT+Argon2 认证 · MySQL 8.0 数据库
  - AI与数据: Flask AI微服务 · Neo4j 图数据库 · SiliconFlow API 大模型调用

#### P07 - 数据源与知识图谱建模

- **Layout**: 左半部分关键数字（803/6/5/10+），右半部分图谱实体关系图
- **Title**: 数据源与知识图谱建模 DATA & KG MODELING
- **Core message**: 803条岗位样本 → 6类实体 → 5种关系，构建职业知识图谱
- **Content**:
  - 关键数字: 803 岗位数据记录 · 6 实体节点类型 · 5 关系类型 · 10+ 数据字段
  - 数据来源: 项目内置招聘岗位样本数据，字段包括职位、城市、公司、学历、经验、技能、薪资、福利、工作描述、行业
  - 图谱实体: Job 岗位 · Company 公司 · City 城市 · Skill 技能 · Degree 学历 · Experience 经验
  - 关系: REQUIRES（需要）· BELONGS_TO（属于）· LOCATED_IN（位于）

---

### Part 2: 核心功能演示 (P08–P14)

#### P08 - Part 02 分隔页

- **Layout**: 全幅绿色背景 + 居中白色大字
- **Title**: PART 02 — 核心功能演示 CORE FEATURES DEMONSTRATION
- **Core message**: 进入系统功能实操演示环节
- **Content**: [分隔页，无额外内容]

#### P09 - 系统入口与主界面

- **Layout**: 上半部分 01-dashboard 截图（主界面宽幅展示），下半部分左侧小图 00-login（登录入口），下方文字说明
- **Title**: 系统入口与主界面 SYSTEM ENTRY & DASHBOARD
- **Core message**: 系统支持注册登录，登录后进入后台管理式主界面，左侧深色导航栏按功能分组
- **Content**:
  - 登录入口: 用户注册/登录，JWT Token认证，角色选择（求职者/企业方）
  - 主界面布局: 左侧深色导航栏（主页、知识图谱、推荐系统、AI问答、简历分析等）· 顶部工具栏 · 中央内容区
  - 注意: 首页统计面板数字为模板展示，尚未接入真实业务统计

#### P10 - 知识图谱可视化

- **Layout**: 左侧文字说明 + 右侧 02-knowledge-graph 截图（截图785×600区域，保留全部UI细节）
- **Title**: 知识图谱可视化 KNOWLEDGE GRAPH VISUALIZATION
- **Core message**: ECharts力导向图渲染岗位-技能-学历-经验-公司-城市六类实体关系网络
- **Content**:
  - 核心特性: ECharts力导向图渲染，支持拖拽交互 · 节点按实体类型着色，直观区分 · 支持按条件筛选子图
  - 图谱说明: 将招聘数据转化为图结构，展示岗位与技能、学历、经验、公司、城市之间的连接关系。相比传统表格，图谱更适合直观解释岗位要求之间的关联。

#### P11 - 职业推荐系统

- **Layout**: 左侧五步流程 + 右侧 03-career-recommendation 截图
- **Title**: 职业推荐系统 JOB RECOMMENDATION SYSTEM
- **Core message**: 用户输入自然语言描述 → NLP提取 → 图谱匹配 → 多维评分 → Top-K推荐，结果表格与关系图谱同步呈现
- **Content**:
  - Step 1 用户输入描述: 输入技能、学历、经验等自然语言
  - Step 2 NLP关键词提取: 技能/学历/经验实体识别
  - Step 3 图谱条件匹配: Neo4j查询满足条件的岗位
  - Step 4 匹配评分计算: 技能命中40% + 学历匹配25% + 经验匹配25% + 薪资匹配10%
  - Step 5 Top-K排序输出: 推荐结果+图谱可视化解释

#### P12 - 职业时间线分析

- **Layout**: 左侧四阶段时间线描述 + 右侧 04-career-timeline 截图
- **Title**: 职业时间线分析 CAREER TIMELINE ANALYSIS
- **Core message**: AI模型根据职业名自动生成四阶段发展路径，时间线组件直观展示成长轨迹
- **Content**:
  - 初级阶段 0-2年: 入门技能掌握，基础项目经验积累
  - 发展阶段 2-5年: 专业技术深化，独立负责项目模块
  - 成熟阶段 5-10年: 团队管理与跨领域协作，技术决策
  - 专家阶段 10年+: 行业影响力建设，技术战略规划
  - 说明: AI生成的职业发展四阶段，包含阶段名称、背景、关键事件和技能要求

#### P13 - AI智能问答

- **Layout**: 左侧六项特性网格 + 右侧 05-ai-chat 截图
- **Title**: AI 智能问答 AI INTELLIGENT Q&A
- **Core message**: 基于DeepSeek-V3大模型，支持自然语言多轮对话，SSE流式响应+Markdown渲染
- **Content**:
  - 自然语言交互 — 用户使用日常语言提问职业发展和技能学习
  - Markdown渲染 — AI回答以Markdown格式输出，支持代码高亮和公式
  - SSE流式响应 — Server-Sent Events实现逐字实时输出
  - 上下文理解 — 支持多轮对话，理解用户意图
  - 职业领域专长 — 针对职业发展、技能学习路径优化
  - 大模型驱动 — 接入DeepSeek-V3，回答质量高

#### P14 - 简历上传与智能分析

- **Layout**: 上半部分 06-resume-upload 截图（上传入口），下半部分 07-resume-analysis 截图（分析结果），中间文字说明处理流程
- **Title**: 简历上传与智能分析 RESUME UPLOAD & ANALYSIS
- **Core message**: 支持PDF/Word/图片多格式上传 → 文本提取 → AI分析 → 职业匹配卡片（职业名/概率/市场/行情）
- **Content**:
  - 上传解析: PDF (PyMuPDF) · Word (python-docx) · 图片 (pytesseract OCR)
  - 分析输出: 职业名称（匹配度最高的方向）· 匹配概率（量化评分）· 市场状况（需求分析）· 发展行情（前景与薪资趋势）
  - 处理链路: 上传简历 → 自动识别格式 → 提取文本内容 → AI模型分析 → 生成职业匹配建议

---

### Part 3: 推荐算法与大模型 (P15–P17)

#### P15 - Part 03 分隔页

- **Layout**: 全幅绿色背景 + 居中白色大字
- **Title**: PART 03 — 推荐算法与大模型 ALGORITHM & LLM INTEGRATION
- **Core message**: 从算法原理到工程实践，解析推荐系统核心
- **Content**: [分隔页，无额外内容]

#### P16 - 推荐算法流程

- **Layout**: 五步水平流程箭头链（chevron chain），右侧权重说明面板
- **Title**: 推荐算法流程 RECOMMENDATION ALGORITHM FLOW
- **Core message**: 自然语言解析 + 知识图谱条件匹配 + Top-K排序 — 基于图谱的可解释推荐
- **Visualization**: chevron_chain_with_tail
- **Content**:
  - Step 01 用户输入自我描述: 输入技能、学历、经验等自然语言描述
  - Step 02 NLP关键词提取: 技能/学历/经验实体识别与提取
  - Step 03 Neo4j图数据库查询: 条件查询与匹配岗位数据
  - Step 04 多维匹配评分计算: 技能命中40% + 学历匹配25% + 经验匹配25% + 薪资匹配10%
  - Step 05 Top-K排序输出: 推荐结果 + 图谱解释可视化
  - 算法特点: ✓ 基于图谱匹配，可解释性强 · ✓ 多维度综合评分 · ✓ Top-K排序，结果精简高效

#### P17 - 大语言模型集成

- **Layout**: 三列功能卡片 — AI职业问答 / 职业时间线 / 简历智能匹配
- **Title**: 大语言模型集成 LLM INTEGRATION
- **Core message**: OpenAI兼容接口接入SiliconFlow平台DeepSeek-V3，不同Prompt模板驱动三大AI功能
- **Content**:
  - AI 职业问答: SSE流式响应 · Markdown渲染 · 多轮对话上下文理解
  - 职业时间线: 自动生成四阶段 · 包含背景和关键事件 · 时间线组件展示
  - 简历智能匹配: 多格式文本提取 · AI职业匹配分析 · 匹配概率+市场行情
  - 技术实现: 通过OpenAI兼容接口接入第三方大模型服务（SiliconFlow平台），调用DeepSeek-V3模型。不同功能模块使用不同Prompt模板，AI返回Markdown由前端渲染。

---

### Part 4: 总结 (P18–P19)

#### P18 - Part 04 分隔页

- **Layout**: 全幅绿色背景 + 居中白色大字
- **Title**: PART 04 — 总结 SUMMARY
- **Core message**: 回顾项目成果与亮点
- **Content**: [分隔页，无额外内容]

#### P19 - 项目亮点与成果

- **Closing impact**: 四大亮点卡片（完整演示闭环 / 可解释推荐 / AI增强分析 / 多维度可视化），让评委记住系统的核心竞争力
- **Layout**: 2×2 四卡片矩阵
- **Title**: 项目亮点与成果 PROJECT HIGHLIGHTS
- **Core message**: 项目形成了完整的本地演示闭环，图谱推荐+AI分析双引擎驱动
- **Content**:
  - ✓ 完整演示闭环: 注册登录 → 知识图谱 → 推荐 → 时间线 → 问答 → 简历分析，全流程本地可运行
  - ✓ 可解释推荐: 基于图谱匹配而非黑盒模型，推荐结果可追溯、可解释，每个推荐都有关系依据
  - ✓ AI增强分析: 大语言模型补充传统图谱查询不足，AI生成职业发展解释、问答和简历分析建议
  - ✓ 多维度可视化: ECharts力导向图 + Element Plus表格/时间线/卡片，多组件协同呈现数据洞察

---

## X. Speaker Notes Requirements

- **Total duration**: 8–10 分钟
- **Notes style**: conversational — 口语化、自然流畅
- **Purpose**: inform + persuade — 展示项目成果，说服评委认可工作质量
- **File naming**: `notes/P01_cover.md` … `notes/P19_highlights.md`

---

## XI. Technical Constraints Reminder

### SVG Generation Must Follow:
1. viewBox: `0 0 1280 720`
2. Background uses `<rect>` elements
3. Text wrapping uses `<tspan>` (`<foreignObject>` FORBIDDEN)
4. Transparency uses `fill-opacity` / `stroke-opacity`; `rgba()` FORBIDDEN
5. FORBIDDEN: `mask`, `<style>`, `class`, `foreignObject`
6. FORBIDDEN: `textPath`, `animate*`, `script`
7. Raw Unicode for typographic symbols; XML reserved chars escaped
8. `clipPath` conditionally allowed only on `<image>` elements

### PPT Compatibility:
- `<g opacity="...">` FORBIDDEN — set on each child individually
- Image transparency uses overlay mask layer
- Inline styles only
