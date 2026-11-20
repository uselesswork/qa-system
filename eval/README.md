# 阶段 14：固定样本评测

`dataset.jsonl` 是五条人工标注的小样本，资料在 `corpus/`。`relevant_sources` 以原始文件名为单位标注，可同时有多个正确来源。样本覆盖直接问答、跨文档、指代追问、无答案和噪声干扰。它只能做回归观察，不代表生产数据分布。

可选择在独立语料上运行真实模型采集预测。此命令会调用 Embedding 和 LLM，产生 API 费用；默认 LLM 是 `gpt-6-luna`，不会使用 Astra：

```powershell
.\.venv\Scripts\python.exe eval\collect_predictions.py --model gpt-6-luna
```

采集时另写入 `eval/predictions.calls.jsonl`（或按 `--output` 推导的同目录 `*.calls.jsonl`；可用 `--calls-output` 指定）。每行对应一次**成功返回**的 LLM 调用，包含唯一 `requestId`、所属数据集 `caseId`、`modelId` 和本次调用 `latencyMs`；模型 ID 优先取供应商响应，响应缺失时回退到请求的模型 ID。`latencyMs` 是非负整数毫秒，四舍五入到最近整数；预测文件的逐题 `latency_ms` 仍保留原来的小数毫秒。仅当供应商在同一次响应中明确返回两项用量时，才同时写入 `inputTokens` / `outputTokens`；零也是有效数值，缺失时两项都省略。可选 `priceVersion` 仅在显式指定时写入。文件不保存问题、答案、Prompt、原文、密钥或异常请求。预测 JSONL 的原字段及评分脚本不变；`model_calls` 仍只是包装器调用次数，不能换算 token。旧基线没有调用明细，不回填 token。

已有同名调用明细时，采集会在读取配置或调用模型前停止，避免覆盖未上报的 `requestId`。要继续上报，请直接使用原文件运行下述 `--retry-calls`；要采集新一轮，请指定新的 `--output` 或 `--calls-output`。默认生成的 `eval/*.calls.jsonl` 已被 Git 忽略，`eval/baselines/` 与测试文件不在该忽略范围；自定义到其他目录的文件应在提交前自行检查。

若 Hub 已有运行，可显式上报。`--hub-run-id` 必须是该 Hub 已存在的运行 UUID；脚本先完成本地采集和逐条落盘，再检查运行并向 `POST /api/runs/{runId}/calls` 发送精简字段。默认不连接 Hub，未传 `--price-version` 时 Hub 会按缺少价格处理。

```powershell
.\.venv\Scripts\python.exe eval\collect_predictions.py --output eval\predictions.jsonl --hub-url http://127.0.0.1:18080 --hub-run-id <已存在运行的UUID>
```

若网络或 Hub 返回错误，明细文件仍在本地。确认运行 ID 后重试，不再调用模型：

```powershell
.\.venv\Scripts\python.exe eval\collect_predictions.py --retry-calls eval\predictions.calls.jsonl --hub-url http://127.0.0.1:18080 --hub-run-id <同一个运行的UUID>
```

重试保留原 `requestId`。Hub 返回 201 计为新增；返回 409 时查询该运行的 `GET /api/runs/{runId}/usage`，只有同一 `requestId` 的模型、整数耗时、成对 token 和价格版本都一致，才计为已送达。找不到、内容冲突或查询失败均报错，原文件保留待人工核对后重试。其他失败也不会删除原文件。旧版含小数 `latencyMs` 的明细不会被悄悄改写或发送，需要先核对并转换为整数，同时保留原 `requestId`。本节的 `--hub-run-id` 是手动指定已有运行的旧入口；报告发布与自动关联入口见下文。Hub 的价格和费用状态以其接口文档为准。

第五阶段增加显式的报告与调用配对发布：`run_eval.py` 评分完成后可调用 Hub 导入现有格式报告，取得 Hub `runId`，再上传 `--calls-file` 中的调用。必须提供实际模型 ID；不要把运行标签当成模型 ID。默认评分仍不连接 Hub。失败后使用原报告与原调用文件执行 `publish_to_hub.py`，不重新调用模型或重新生成报告哈希：

```powershell
.\.venv\Scripts\python.exe eval\run_eval.py --predictions eval\predictions.jsonl --output eval\reports\run.report.json --run-label example --hub-url http://127.0.0.1:8080 --calls-file eval\predictions.calls.jsonl --model-id example-model
.\.venv\Scripts\python.exe eval\publish_to_hub.py --report eval\reports\run.report.json --calls-file eval\predictions.calls.jsonl --hub-url http://127.0.0.1:8080 --model-id example-model
```

发布前需人工确认报告、预测与调用明细来自同一次采集。相同报告和相同调用重复发布会使用原 Hub 运行并去重；本轮只用已有五题报告和一条**合成调用**在一次性隔离 Hub/MySQL 验证，未运行真实模型。尚无后台自动调度或真实模型全链路验收。

2026-11-20 用假 LLM 响应和 `httpx.MockTransport` 验证了小数耗时转整数、成对用量、409 一致与冲突、查询失败、同名文件保护及重试模式不采集；相关测试 **17 passed**，完整无密钥回归 **76 passed**。本轮未运行真实模型，也未对真实 Hub 服务做联调。

评测脚本读取**同一次系统运行**产生的 `predictions.jsonl`，每条记录格式：

```json
{"id":"direct","retrieved_sources":["project.md"],"cited_sources":["project.md"],"answer":"2026年6月18日","refused":false,"latency_ms":120.0,"model_calls":2}
```

`retrieved_sources` 按召回排名填写，即使最终没有引用也要记录；`cited_sources` 填最终展示给用户的来源。无答案时 `refused=true`。多轮样本需按 `history` 先建立上下文，记录最后一问的预测。每个 `id` 必须有一条预测；漏项或重复项直接报错。已有页面接口只返回最终引用，不返回完整召回排名，因此不能只把页面历史当作完整预测文件；要在评测运行中另行采集召回顺序。

```powershell
.\.venv\Scripts\python.exe eval\run_eval.py --predictions eval\predictions.jsonl --output eval\reports\run.json --run-label luna-20261116
```

报告记录样本数、`k`、运行标签、UTC 时间、Python 版本、输入文件 SHA-256，以及逐条结果。`Hit@k` 和 `MRR@k` 仅统计有标准证据的样本；引用准确率是每条可回答样本的“正确引用数 ÷ 引用总数”的平均值；无答案正确率只统计无答案样本。答案包含率是简单字符串核对，不等于语义正确率。报告中的延迟和模型调用次数完全取自预测文件，脚本不会伪造它们。

## 首次真实模型基线（2026-11-17）

已使用 `gpt-6-luna`、默认服务端温度、`text-embedding-3-small`、`top_k=5`、当前 `eval/corpus/` 和 `dataset.jsonl` 采集一次预测。原始预测与评分报告分别保存在 [`baselines/2026-11-17-gpt-6-luna.predictions.jsonl`](baselines/2026-11-17-gpt-6-luna.predictions.jsonl) 和 [`baselines/2026-11-17-gpt-6-luna.report.json`](baselines/2026-11-17-gpt-6-luna.report.json)。报告记录数据集与预测文件 SHA-256，可用于后续同配置对照。

本次 5 条案例中，4 条可回答题的文档级 `Hit@5=1.0`、`MRR@5=1.0`、引用准确率 `1.0`、答案包含率 `1.0`；1 条无答案题正确拒答。延迟 P50 为 `4659.872 ms`、P95 为 `6710.773 ms`，共 10 次 LLM 调用。语料只有 3 份短文档，命中率不能外推到真实用户资料；答案包含率只检查字符串，未人工评判语义质量。采集脚本直接运行现有 RAG 组件、真实 Embedding 与 LLM，向量存在内存中；这份基线不包含 MySQL、Qdrant、Redis、HTTP 和浏览器耗时，浏览器端到端另有阶段 11～12 的验收记录。

复跑时保持模型和语料不变，先重新采集，再对新的预测文件运行 `run_eval.py`；不要把不同配置的结果直接当成改造前后效果。

## 阶段 16 扩展样本

独立的 [`stage16/dataset.jsonl`](stage16/dataset.jsonl) 和 [`stage16/corpus/`](stage16/corpus/) 增加相似项目、过时草案、不同日期、无答案及恶意指令案例。`answer_contains` 会忽略空白字符，`answer_excludes` 可列出不允许出现的旧日期或错误日期。采集脚本可选择临时本地 Qdrant，运行结束即删除临时集合；不会写入用户知识库：

```powershell
.\.venv\Scripts\python.exe eval\collect_predictions.py --dataset eval\stage16\dataset.jsonl --corpus eval\stage16\corpus --vector-backend qdrant-local --model gpt-6-luna --output eval\predictions.jsonl
.\.venv\Scripts\python.exe eval\run_eval.py --dataset eval\stage16\dataset.jsonl --corpus eval\stage16\corpus --predictions eval\predictions.jsonl --output eval\reports\stage16.json --run-label stage16-qdrant-luna
```

首次真实预测及其报告保存在 [`baselines/2026-11-18-stage16-qdrant-luna.predictions.jsonl`](baselines/2026-11-18-stage16-qdrant-luna.predictions.jsonl) 和 [`baselines/2026-11-18-stage16-qdrant-luna.report.json`](baselines/2026-11-18-stage16-qdrant-luna.report.json)。旧五题在同模型、同内存后端、同语料下的复跑报告为 [`baselines/2026-11-18-stage14-regression-luna.report.json`](baselines/2026-11-18-stage14-regression-luna.report.json)。详见[阶段 16 说明](../docs/第十六阶段-检索质量与安全加固.md)。
