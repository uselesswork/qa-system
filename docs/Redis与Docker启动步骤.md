# Windows 上启动 Redis、Qdrant 和异步索引

> [返回 README](../README.md) · [阶段 13 说明](第十三阶段-异步索引与任务状态.md)

本项目的异步索引需要 Redis 作 Celery 队列、独立 Qdrant 存储向量，还需要 MySQL、Web、Celery worker 和 beat。以下命令均在项目根目录 `D:\qa-system` 的 **PowerShell** 中执行。先确保 Docker Desktop 已安装；安装与首次启动方式见 [Docker 官方 Windows 文档](https://docs.docker.com/desktop/setup/install/windows-install/)。

## 1. 启动 Docker Desktop

从 Windows 开始菜单打开 **Docker Desktop**，完成首次启动界面并等待引擎显示运行中。然后在 PowerShell 检查：

```powershell
cd D:\qa-system
docker version
docker compose version
```

`docker version` 应同时显示 Client 和 Server。若只有 Client，或出现 `dockerDesktopLinuxEngine` 管道不存在，说明 Docker 引擎还没就绪；先回到 Docker Desktop 查看启动状态。单独运行 `docker.exe` 不会启动引擎。

## 2. 启动 Redis 与 Qdrant

项目根目录的 `compose.ingestion.yaml` 定义 Redis `6379` 和 Qdrant `6333`，只绑定本机 `127.0.0.1`，数据保存在 Docker 命名卷中。执行：

```powershell
docker compose -f compose.ingestion.yaml config --quiet
docker compose -f compose.ingestion.yaml up -d
docker compose -f compose.ingestion.yaml ps
```

`ps` 中两个服务都应为运行状态。`up -d` 在后台启动容器，详见 [Docker Compose 文档](https://docs.docker.com/reference/cli/docker/compose/up/)。首次运行需要拉取镜像，耗时取决于网络。

检查 Redis 和 Qdrant：

```powershell
docker compose -f compose.ingestion.yaml exec -T redis redis-cli ping
Invoke-RestMethod http://127.0.0.1:6333/collections
```

Redis 应返回 `PONG`，这是 [Redis 官方的连接检查命令](https://redis.io/docs/latest/commands/ping/)；Qdrant 应返回包含 `collections` 的 JSON。这里使用**容器内**的 `redis-cli`。本机现有 `D:\redis\redis-cli.exe` 曾出现 Windows DLL 重定位错误，无需依赖它。

若服务没起来，分别查看日志：

```powershell
docker compose -f compose.ingestion.yaml logs --tail=80 redis
docker compose -f compose.ingestion.yaml logs --tail=80 qdrant
```

## 3. 配置项目并升级数据库

已有 `.env` 时在其中添加或修改以下两行；不要用 `.env.example` 覆盖现有密钥和数据库密码。两行**都配置**才启用异步索引：

```dotenv
REDIS_URL=redis://127.0.0.1:6379/0
QDRANT_URL=http://127.0.0.1:6333
```

Web 与 worker 应使用同一份 `.env`。安装阶段 13 依赖并升级 MySQL：

```powershell
.\.venv\Scripts\python.exe -m pip install "celery[redis]>=5.5,<5.7" "packaging<26"
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m alembic current
```

`alembic current` 应显示 `0005_ingestion_tasks`。本机数据库此前已迁移；换电脑时仍需运行升级命令。

## 4. 分别启动 Web、worker 和 beat

打开三个 PowerShell 终端，均先 `cd D:\qa-system`，各运行一条：

```powershell
# 终端 1：Web
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8090
```

```powershell
# 终端 2：Celery worker
.\.venv\Scripts\celery.exe -A src.tasks.ingest:celery_app worker --pool=solo --loglevel=info
```

```powershell
# 终端 3：Celery beat，负责补发排队任务与恢复超时任务
.\.venv\Scripts\celery.exe -A src.tasks.ingest:celery_app beat --loglevel=info
```

Windows 开发环境使用 `--pool=solo`；[Celery 并发文档](https://docs.celeryq.dev/en/stable/userguide/concurrency/index.html)说明了 Windows 上进程池的限制和 `solo` 选项。打开 <http://127.0.0.1:8090/>，上传一份小型 Markdown：异步模式下上传接口应返回 `202` 与任务 ID，页面依次显示“待索引 / 索引中 / 已索引”。仅 Docker 容器运行、worker 未运行时任务会停在“待索引”。

## 旧资料与停止服务

切换到独立 Qdrant 后，原 `data/qdrant/` 的向量**不会自动迁入**容器。登录后对已有“已索引”文档点击“重建索引”，等待重新完成，再测试问答。请不要直接删除旧数据目录。

停止 Web、worker、beat 时在各自终端按 `Ctrl+C`。停止两个容器但保留数据卷：

```powershell
docker compose -f compose.ingestion.yaml stop
```

下次重新执行 `docker compose -f compose.ingestion.yaml up -d`。如果不再需要这组容器，可用 `docker compose -f compose.ingestion.yaml down` 移除容器与网络；**不要加 `-v`**，否则会删除 Redis 和 Qdrant 数据卷。
