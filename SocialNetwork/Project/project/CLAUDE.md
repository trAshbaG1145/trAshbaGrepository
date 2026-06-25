# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

智能职位推荐系统 — 面向"社交网络数据分析与处理"课程项目。三部分协作：Spring Boot 后端 + Vue 3 前端 + Flask AI 服务。

## 仓库结构

```
project/
├── JobRec_Back/       # Java Spring Boot 后端 (端口 8090)
├── JobRec_Front/      # Vue 3 前端 (端口 8089) + Flask AI 服务 (端口 8080)
│   └── Backend/       # Python Flask AI/推荐引擎
└── reference/         # 课程参考材料（推荐系统实验等）
```

## 后端 (JobRec_Back)

Spring Boot 3.4.5 + MyBatis-Plus 3.5.5，Java 17，Maven。

### 启动

```bash
cd JobRec_Back
./mvnw spring-boot:run    # 端口 8090
```

### 运行测试

```bash
cd JobRec_Back
./mvnw test
./mvnw test -Dtest=UserServiceTest   # 单个测试类
```

### 数据库

MySQL 8.0+，数据库名 `jobrec`。用 `jobrec.sql` 初始化表结构和种子数据。配置在 `src/main/resources/application.yml` — 启动前需按本地环境修改数据库连接、Redis 连接和文件路径配置。

### 架构分层

```
controller/   # REST API — UserController, FirmController, JobCardController, CandidateController, SeekCardController, MenuController, UploadController
service/      # 业务逻辑接口 + service/Impl/ 实现
mapper/       # MyBatis-Plus BaseMapper
entity/       # 数据库实体 (Candidate, Firm, JobCard, SeekerCard, Menu, Role, UserRole 等)
dto/          # 数据传输对象 (LoginDTO, CandidateDTO, JobCardDTO 等)
config/       # Spring 配置 — CorsConfig, WebConfig (JWT 拦截器注册), MPConfig, RedisConfig
interceptor/  # JWT Token 验证拦截器 (LoginInterceptor — /user/** 路径免拦截)
exception/    # CustomException + GlobalExceptionHandler
common/       # Result<T> 统一响应封装
```

**认证**：JWT (io.jsonwebtoken/jjwt 0.9.1)，密码哈希用 Argon2。登录拦截器对 `/user/login`、`/user/register` 等用户端点放行。前端在 `Authorization` header 传 token，后端 LoginInterceptor 校验。

**技术栈**：Druid 连接池、Redis (Jedis + Spring Data Redis)、Lombok、Swagger annotations、Spring Validation。

## 前端 (JobRec_Front)

Vue 3 + Vite 6 + Element Plus 2.9 + Pinia 3 + ECharts 5。

### 启动

```bash
cd JobRec_Front
npm install
npm run dev          # 端口 8089，API 代理到 127.0.0.1:8080
```

### 构建

```bash
npm run build
```

### 架构

```
src/
├── views/Home.vue              # 主布局 (含 Dashboard)
├── components/
│   ├── Login/                  # 登录、角色选择 (Login.vue, Select.vue, Role.vue, Role2.vue)
│   ├── Compo/                  # 通用组件 (404, Card1-3, Loader, Loading, Mouse, Particles)
│   ├── Card/                   # 用户卡片 (Candidate.vue, Firm.vue)
│   ├── Person/                 # 个人中心 (userInfo, firmInfo, resumeInfo, Space, Avatar, restPwd)
│   ├── Dashboard.vue, JobDetail.vue, KG.vue, Chat.vue, Resume.vue, Recommend.vue, TimeLine.vue, Logo.vue
├── router/index.js             # 动态路由 — 登录后从 localStorage menus 构建路由
├── stores/index.js             # Pinia loading store
└── utils/axios.js              # Axios 实例 — 自动附加 JWT token，401 时跳转登录
```

**动态路由**：`setRouters()` 从 `localStorage.getItem('menus')` 读取菜单 JSON，动态 `router.addRoute()` 注入受保护路由。组件路径按 `/components/<component>.vue` 解析。

**API 代理**：`vite.config.js` 中 `/api` 代理到 `http://127.0.0.1:8080`（Flask AI 服务）。

## Flask AI 服务 (JobRec_Front/Backend)

Python Flask 应用（端口 8080），调用 SiliconFlow API (DeepSeek-V3) 提供 AI 功能。

### 启动

```bash
cd JobRec_Front/Backend
D:\anaconda3\python.exe app.py     # 端口 8080
```

先安装依赖：`D:\anaconda3\python.exe -m pip install flask flask-cors openai`（以及其他缺少的包如 `pandas`, `scikit-learn` 等）。

### API 端点

| 端点 | 功能 |
|------|------|
| `/api/recommend` | 根据技能/学历/期望薪资推荐职业 |
| `/api/search` | 按职位名搜索 |
| `/api/upload` | 上传简历 PDF/Word/图片，提取文本并匹配职业 |
| `/api/ask` | AI 职业问答 (SSE 流式) |
| `/api/analyze` | 职业发展时间线分析 |

### 关键模块

```
Backend/
├── app.py                      # Flask 主入口
├── function/
│   ├── job_rec.py              # JobRecommender — 调用 LLM 分析简历
│   ├── resume_parser.py        # CareerCounselor — AI 问答
│   ├── career_analyzer.py      # CareerTimelineAnalyzer — 职业时间线
│   └── model_1.py              # 推荐模型
├── KG_processing.py            # 知识图谱句子处理
├── KG_search.py                # 知识图谱搜索
├── KG_answer.py                # 知识图谱推荐
└── upload.py                   # 文件上传 + PDF/Word/图片文本提取
```

## 安全注意事项

- `JobRec_Front/Backend/app.py` 第 20 行硬编码了 SiliconFlow API key — 应迁移到环境变量
- `JobRec_Back/src/main/resources/application.yml` 硬编码了数据库密码和本地路径 — 启动前需修改
- 数据库种子数据中包含测试用户密码哈希

## 参考项目 (reference/)

`reference/MF_Rec/main.py` — 矩阵分解 (Matrix Factorization) 协同过滤推荐算法，使用 MovieLens 1M 格式数据。可作为推荐算法参考。
