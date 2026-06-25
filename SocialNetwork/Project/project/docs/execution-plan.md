# 执行计划：职位推荐系统复现

> 最后更新：2026-06-25

## 当前状态：全部完成 ✅

## Phase 1：环境准备 ✅

| 步骤 | 操作 | 结果 |
|------|------|------|
| 1.1 | 安装 JDK 17 (winget Microsoft.OpenJDK.17) | `C:/Program Files/Microsoft/jdk-17.0.19.10-hotspot` |
| 1.2 | Docker 启动 Redis 7 | `docker exec redis redis-cli ping` → PONG |
| 1.3 | 创建 MySQL 数据库 jobrec | `CREATE DATABASE IF NOT EXISTS jobrec` |
| 1.4 | 导入种子数据 | `mysql -u root -p jobrec < jobrec.sql` |
| 1.5 | Docker 启动 Neo4j 5 | `bolt://localhost:7687` auth: neo4j/TYH041113 |
| 1.6 | Node.js 环境已有 (v22.16.0) | — |
| 1.7 | Python 3.12 已有 + 补装依赖 (flask-cors, py2neo, pytesseract, pdfplumber, python-docx, Pillow) | — |

## Phase 2：安全处理 ⚠️

| 步骤 | 操作 | 状态 |
|------|------|------|
| 2.1 | 提取 API key 到环境变量 | **推迟**（用户决定先用原 key 跑通，之后替换） |
| 2.2 | application.yml 密码已更新为本地凭据 | ✅ |
| 2.3 | `.env` 文件创建 (`VITE_APP_BASE_API=http://localhost:8090`) | ✅ |

> **遗留**：`app.py` 第 20 行仍含硬编码 SiliconFlow API key，`application.yml` 含数据库密码。生产环境需迁移。

## Phase 3：配置适配 ✅

| 步骤 | 操作 | 结果 |
|------|------|------|
| 3.1 | MySQL 密码改为 `MySQL@999999`，端口改为 `3413` | 连接成功 |
| 3.2 | 文件上传路径改为相对路径 `uploaded-files/` | 已创建目录 |
| 3.3 | pom.xml source/target 从 21 改为 17（匹配本地 JDK） | 编译通过 |
| 3.4 | 创建 `JobRec_Front/.env` | VITE_APP_BASE_API 指向 8090 |

## Phase 4：构建与启动 ✅

| 步骤 | 操作 | 结果 |
|------|------|------|
| 4.1 | `./mvnw compile` | 通过 |
| 4.2 | `./mvnw spring-boot:run` → 端口 8090 | 运行中 |
| 4.3 | `python app.py` → 端口 8080 (Flask) | 运行中 |
| 4.4 | `npm install` | 通过 (有 npm audit 警告) |
| 4.5 | `npm run dev` → 端口 8089 (Vite) | 运行中 |

## Phase 5：端到端验证 ✅

| 步骤 | 操作 | 结果 |
|------|------|------|
| 5.1 | `curl localhost:8089` | 200，页面正常 |
| 5.2 | `POST /user/login` (Spring Boot) | 200，数据库链路正常 |
| 5.3 | `POST /user/register` (Spring Boot) | 200，注册成功并返回 JWT + 菜单 |
| 5.4 | `POST /api/search` (Flask + Neo4j) | 200，KG 搜索返回岗位 + 图谱数据 |
| 5.5 | `POST /api/recommend` (Flask + Neo4j) | 200，推荐岗位 + ECharts 关系图 |
| 5.6 | `POST /api/analyze` (Flask + LLM) | 200，AI 生成职业时间线 |
| 5.7 | 浏览器访问前端 | 页面加载正常，菜单动态路由可用 |

## Phase 6：Neo4j 数据导入 ✅

| 步骤 | 操作 | 结果 |
|------|------|------|
| 6.1 | 创建 `neo4j_import.py` 导入脚本 | 从 MySQL jobSys 读取 803 条岗位 |
| 6.2 | 运行导入 | Neo4j 创建 734 Job, 751 Company, 2048 Skill, 8 Degree, 7 Experience 节点及关系 |

## 已解决的历史遗留问题

| # | 问题 | 解决方法 |
|---|------|----------|
| 1 | 种子账号密码未知 | 通过 `/user/register` API 注册新用户（需 `scope` 和 `email` 字段） |
| 3 | Neo4j 知识图谱无数据 | 创建 `neo4j_import.py` 导入脚本，从 jobSys 表批量导入 |

## 服务总览

```
Docker Compose:
  mysql:8.0          → localhost:3413 (容器 3306)
  redis:7-alpine     → localhost:6380 (容器 6379)
  neo4j:community    → localhost:7688 Bolt / 7475 HTTP (容器 7687/7474)

本地进程:
  Flask (python)     → localhost:8081  /api/*
  Spring Boot (java) → localhost:8090  其他所有 API
  Vite (node)        → localhost:8089  前端页面
```

**注意**：端口映射与 docker-compose.yml 保持一致。application.yml 中 MySQL 端口为 3413，Redis 端口为 6380，kg_config.py 中 Neo4j Bolt 端口为 7688。

## 启动命令速查

```bash
# 方式一：一键启动（推荐）
cd project
start-project.bat

# 方式二：手动按顺序启动
docker compose up -d  # 启动基础设施
# 等待容器 healthy

# Neo4j 数据导入（首次或数据重置后）
cd JobRec_Front/Backend
python neo4j_import.py

# Flask (终端1)
cd JobRec_Front/Backend && python app.py

# Spring Boot (终端2)
export JAVA_HOME="C:/Program Files/Java/jdk-21"
cd JobRec_Back && ./mvnw spring-boot:run

# 前端 (终端3)
cd JobRec_Front && npm run dev

# 浏览器访问 http://localhost:8089
```
