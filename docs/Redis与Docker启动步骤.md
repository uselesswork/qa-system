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

项目根目录的 `compose.ingestion.yaml` 默认将容器内 Redis `6379` 映射到宿主机 `127.0.0.1:6379`，Qdrant 使用 `127.0.0.1:6333`；数据保存在 Docker 命名卷中。若启动时报 `127.0.0.1:6379 ... bind ... forbidden by its access permissions`，先检查本机 `6379` 是否已被占用：

```powershell
Get-NetTCPConnection -LocalPort 6379 -State Listen -ErrorAction SilentlyContinue
Get-Service Redis -ErrorAction SilentlyContinue
```

如果本机已有 Windows Redis 监听 `6379`，可以只在**这台电脑**被 Git 忽略的 `.env` 中新增 `REDIS_HOST_PORT=16379`。Compose 会将 Docker Redis 发布到 `127.0.0.1:16379`；其他电脑不设置该变量，仍默认使用 `6379`。不要停止原生 Redis 或删除数据卷来处理此冲突。设置后照常执行：

```powershell
docker compose -f compose.ingestion.yaml config --quiet
docker compose -f compose.ingestion.yaml up -d
docker compose -f compose.ingestion.yaml ps
```

`ps` 中两个服务都应为运行状态；端口冲突的电脑应显示 `127.0.0.1:16379->6379/tcp`，其他电脑显示 `127.0.0.1:6379->6379/tcp`。`up -d` 在后台启动容器，详见 [Docker Compose 文档](https://docs.docker.com/reference/cli/docker/compose/up/)。首次运行需要拉取镜像，耗时取决于网络。

检查 Redis 和 Qdrant：

```powershell
docker compose -f compose.ingestion.yaml exec -T redis redis-cli ping
Invoke-RestMethod http://127.0.0.1:6333/collections
```

Redis 应返回 `PONG`，这是 [Redis 官方的连接检查命令](https://redis.io/docs/latest/commands/ping/)；Qdrant 应返回包含 `collections` 的 JSON。这里使用**容器内**的 `redis-cli`。本机现有 `D:\redis\redis-cli.exe` 曾出现 Windows DLL 重定位错误，无需依赖它。

如果浏览器删除文档提示“删除向量失败”，但 Qdrant 容器仍为 `Up`，可先检查本机 Qdrant HTTP 是否返回 200。Windows 系统代理可能让 Python HTTP 客户端访问 `127.0.0.1:6333` 时收到 502；项目对本机 Qdrant 地址已禁用客户端的环境代理。更新代码后需重启 Web 与 worker，再在页面点击“删除文档”重试。

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

`REDIS_HOST_PORT` 只决定 Docker Redis 对本机开放哪个端口，不会修改应用的 `REDIS_URL`。如果应用要使用端口为 `16379` 的 Docker Redis，需将本机 `.env` 的 `REDIS_URL` 也设为 `redis://127.0.0.1:16379/0`，并重启 Web、worker、beat。如果原生 Redis `6379` 已有 Celery 队列消息，切换地址不会自动迁移这些消息；先确认旧队列任务如何处理。只为解决 Docker 启动冲突时，可以保留 `REDIS_URL=redis://127.0.0.1:6379/0`，应用继续使用原生 Redis，Docker Redis 虽启动但不会被应用使用。2026-11-17 这台电脑采用后一种方式，旧队列未迁移。

Web 与 worker 应使用同一份 `.env`。安装阶段 13 依赖并升级 MySQL：

```powershell
.\.venv\Scripts\python.exe -m pip install "celery[redis]>=5.5,<5.7" "packaging<26"
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m alembic current
```

`alembic current` 应显示 `0005_ingestion_tasks`。本机数据库此前已迁移；换电脑时仍需运行升级命令。

## 4. 分别启动 Web、worker 和 beat

### 一条命令启动（推荐）

先启动 Docker Desktop 并确认 MySQL 正在运行。进入项目根目录后执行：

```powershell
.\run.cmd
```

脚本会启动 Redis、Qdrant 容器，并在后台启动 Web、worker、beat，无需打开三个终端。第一次启动前仍需按第 3 节配置 `.env`、安装依赖并执行数据库迁移。查看状态或停止：

```powershell
.\run.cmd status
.\run.cmd stop
```

日志保存在被 Git 忽略的 `.runtime/`；启动失败时先查看对应的 `web.err.log`、`worker.err.log` 或 `beat.err.log`。重复运行 `start` 不会重复启动由脚本管理的进程。切换到脚本前，先在原来三个终端按 `Ctrl+C` 结束手动进程；脚本只会停止自己启动的进程。`stop` 还会停止这组 Redis、Qdrant 容器，但保留 Docker 数据卷。

### 手动启动（排查问题时使用）

打开三个 PowerShell 终端，均先 `cd D:\qa-system`，各运行一条：

```powershell
# 终端 1：Web
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8090
```

```powershell
# 终端 2：Celery worker，负责处理新任务
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
