# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

智能职位推荐系统 — 面向"社交网络数据分析与处理"课程项目。三部分协作：Spring Boot 后端 + Vue 3 前端 + Flask AI 服务。

## 仓库结构

```
project/
├── JobRec_Back/       # Java Spring Boot 后端 (端口 8090)
├── JobRec_Front/      # Vue 3 前端 (端口 8089) + Flask AI 服务 (端口 8081)
│   └── Backend/       # Python Flask AI/推荐引擎
├── docs/              # 项目文档和演示截图
├── reference/         # 课程参考材料（推荐系统实验等）
├── docker-compose.yml # Docker 基础设施编排 (MySQL, Redis, Neo4j)
├── start-project.bat  # Windows 一键启动脚本
└── start-project.ps1  # PowerShell 一键启动脚本
```

## 一键启动

项目根目录提供了启动脚本，自动拉起 Docker 基础设施和所有服务：

```bash
# Windows 命令行
start-project.bat

# 或直接运行 PowerShell 脚本
powershell -ExecutionPolicy Bypass -File start-project.ps1
```

脚本流程：
1. `docker compose up -d` → 启动 MySQL (3413)、Redis (6380)、Neo4j (7688/7475)
2. 等待容器健康检查通过
3. 依次启动 Spring Boot (8090)、Flask (8081)、Vite (8089)
4. 验证所有端口后自动打开浏览器

## 基础设施 (Docker Compose)

```bash
# 启动数据库和中间件
docker compose up -d

# 查看状态
docker compose ps
```

| 服务 | Docker 端口 | 容器内端口 | 凭据 |
|------|------------|-----------|------|
| MySQL 8.0 | 3413 | 3306 | root / MySQL@999999 |
| Redis 7 | 6380 | 6379 | 无密码 |
| Neo4j Community | 7688 (Bolt) / 7475 (HTTP) | 7687 / 7474 | neo4j / TYH041113 |

**注意**：首次启动 MySQL 容器时会自动执行 `jobrec.sql` 初始化表结构和种子数据。Neo4j 容器需要额外导入知识图谱数据（见下方）。

### 导入 Neo4j 知识图谱数据

Docker Compose 启动后，Neo4j 为空库，需运行导入脚本：

```bash
cd JobRec_Front/Backend
python neo4j_import.py
```

该脚本从 MySQL `jobSys` 表读取 803 条岗位数据，在 Neo4j 中创建 Job、Company、City、Skill、Degree、Experience 节点及关系。

## 后端 (JobRec_Back)

Spring Boot 3.4.5 + MyBatis-Plus 3.5.5，Java 17，Maven。

### 启动

```bash
cd JobRec_Back
./mvnw spring-boot:run    # 端口 8090
```

**注意**：MySQL、Redis、Neo4j 需通过 Docker Compose 先行启动（见下方"一键启动"）。

### 运行测试

```bash
cd JobRec_Back
./mvnw test
./mvnw test -Dtest=UserServiceTest   # 单个测试类
```

### 数据库

MySQL 8.0+，数据库名 `jobrec`。用 `jobrec.sql` 初始化表结构和种子数据。

**Docker Compose 端口映射**：MySQL 容器 3306 → 宿主机 3413。配置在 `src/main/resources/application.yml` — 使用 Docker 时无需修改，使用本地 MySQL 时需修改端口为 3306。

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
npm run dev          # 端口 8089，API 代理到 127.0.0.1:8081
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

**API 代理**：`vite.config.js` 中 `/api` 代理到 `http://127.0.0.1:8081`（Flask AI 服务）。

**登录注册**：注册接口 `/user/register` 需要 `scope` 字段（1=求职者, 2=企业方）和 `email` 字段。登录接口 `/user/login` 只需 `username` 和 `password`。

## Flask AI 服务 (JobRec_Front/Backend)

Python Flask 应用（端口 8081，通过 `FLASK_PORT` 环境变量配置），调用 SiliconFlow API (DeepSeek-V3) 提供 AI 功能。

### 启动

```bash
cd JobRec_Front/Backend
pip install -r requirements.txt    # 安装依赖
python app.py                      # 端口 8081 (默认)
```

或指定端口：`$env:FLASK_PORT='8081'; python app.py`

### API 端点

| 端点 | 功能 |
|------|------|
| `/api/recommend` | 根据技能/学历/期望薪资推荐职业 (KG + Neo4j) |
| `/api/search` | 按职位名和条件搜索 (KG + Neo4j) |
| `/api/upload` | 上传简历 PDF/Word/图片，提取文本并匹配职业 (LLM) |
| `/api/ask` | AI 职业问答 (SSE 流式, LLM) |
| `/api/analyze` | 职业发展时间线分析 (LLM) |

### 关键模块

```
Backend/
├── app.py                      # Flask 主入口
├── kg_config.py                # Neo4j 连接配置 (bolt://localhost:7688)
├── neo4j_import.py             # Neo4j 数据导入脚本 (从 MySQL 导入岗位)
├── function/
│   ├── job_rec.py              # JobRecommender — 调用 LLM 分析简历
│   ├── resume_parser.py        # CareerCounselor — AI 问答 (SSE)
│   ├── career_analyzer.py      # CareerTimelineAnalyzer — 职业时间线
│   ├── model_1.py              # 推荐模型
│   ├── demo.py                 # MySQL 数据导入 (Excel → jobSys 表)
│   └── API.py                  # 简历 OCR/摘要 (TrOCR + BART)
├── KG_processing.py            # 用户输入处理，提取技能和学历
├── KG_search.py                # 知识图谱职位搜索 (Neo4j Cypher)
├── KG_answer.py                # 知识图谱职位推荐 (Neo4j Cypher)
└── upload.py                   # 文件上传 + PDF/Word/图片文本提取
```

## 安全注意事项

- `JobRec_Front/Backend/app.py` 第 20 行硬编码了 SiliconFlow API key — 应迁移到环境变量
- `JobRec_Back/src/main/resources/application.yml` 硬编码了数据库密码 — Docker Compose 使用时无需修改，部署时需迁移到环境变量
- 数据库种子数据中包含测试用户密码哈希（Argon2 加密）
- Neo4j 默认密码 `TYH041113` 在 `kg_config.py` 和 `docker-compose.yml` 中均有明文 — 生产环境需修改
- JWT 签名密钥 `yuhang` 写死在 `JwtUtil.java` 中 — 应改为环境变量注入

## 参考项目 (reference/)

`reference/MF_Rec/main.py` — 矩阵分解 (Matrix Factorization) 协同过滤推荐算法，使用 MovieLens 1M 格式数据。可作为推荐算法参考。
