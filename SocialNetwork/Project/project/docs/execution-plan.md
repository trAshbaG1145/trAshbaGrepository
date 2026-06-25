# 执行计划：职位推荐系统复现

> 最后更新：2026-05-07 23:10

## 当前状态：Phase 4 完成，Phase 5 部分完成

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

## Phase 5：端到端验证 ⚠️

| 步骤 | 操作 | 结果 |
|------|------|------|
| 5.1 | `curl localhost:8089` | 200，页面正常 |
| 5.2 | `POST /user/login` (Spring Boot) | 200，数据库链路正常 |
| 5.3 | `POST /api/search` (Flask) | 200，KG + AI 响应正常 |
| 5.4 | `POST /api/recommend` (Flask) | 200，但 SiliconFlow API 调用耗时长 |
| 5.5 | 浏览器登录 | **未完成**（种子账号密码是 Argon2 哈希，无法反推） |

## 遗留问题

| # | 问题 | 优先级 | 处理方向 |
|---|------|--------|----------|
| 1 | **种子账号密码未知** | 高 | 注册新账号或通过 API 直接注册 |
| 2 | **SiliconFlow API Key** | 高 | 用户后续替换为自有 key |
| 3 | **Neo4j 知识图谱无数据** | 中 | 需原始数据导入脚本 |
| 4 | **Redis 无预热数据** | 低 | 不影响功能 |

## 服务总览

```
Docker:
  redis:7-alpine     → localhost:6379
  neo4j:5-community  → localhost:7474 (HTTP) / 7687 (Bolt)

本地进程:
  Flask (python)     → localhost:8080  /api/*
  Spring Boot (java) → localhost:8090  其他所有 API
  Vite (node)        → localhost:8089  前端页面
```

## 启动命令速查

```bash
# 按顺序启动所有服务（每次需要复现时）
docker start redis neo4j

# Flask (终端1)
cd JobRec_Front/Backend && D:/anaconda3/python.exe app.py

# Spring Boot (终端2)
export JAVA_HOME="C:/Program Files/Microsoft/jdk-17.0.19.10-hotspot"
cd JobRec_Back && ./mvnw spring-boot:run

# 前端 (终端3)
cd JobRec_Front && npm run dev

# 浏览器访问 http://localhost:8089
```
