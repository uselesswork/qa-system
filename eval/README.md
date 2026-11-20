# 阶段 14：固定样本评测

`dataset.jsonl` 是五条人工标注的小样本，资料在 `corpus/`。`relevant_sources` 以原始文件名为单位标注，可同时有多个正确来源。样本覆盖直接问答、跨文档、指代追问、无答案和噪声干扰。它只能做回归观察，不代表生产数据分布。

可选择在独立语料上运行真实模型采集预测。此命令会调用 Embedding 和 LLM，产生 API 费用；默认 LLM 是 `gpt-6-luna`，不会使用 Astra：

```powershell
.\.venv\Scripts\python.exe eval\collect_predictions.py --model gpt-6-luna
```

采集时另写入 `eval/predictions.calls.jsonl`（或按 `--output` 推导的同目录 `*.calls.jsonl`；可用 `--calls-output` 指定）。每行对应一次**成功返回**的 LLM 调用，包含唯一 `requestId`、所属数据集 `caseId`、`modelId` 和本次调用 `latencyMs`；模型 ID 优先取供应商响应，响应缺失时回退到请求的模型 ID。仅当供应商在同一次响应中明确返回两项用量时，才同时写入 `inputTokens` / `outputTokens`；零也是有效数值，缺失时两项都省略。可选 `priceVersion` 仅在显式指定时写入。文件不保存问题、答案、Prompt、原文、密钥或异常请求。预测 JSONL 的原字段及评分脚本不变；`model_calls` 仍只是包装器调用次数，不能换算 token。旧基线没有调用明细，不回填 token。

若 Hub 已有运行，可显式上报。`--hub-run-id` 必须是该 Hub 已存在的运行 UUID；脚本先完成本地采集和逐条落盘，再检查运行并向 `POST /api/runs/{runId}/calls` 发送精简字段。默认不连接 Hub，未传 `--price-version` 时 Hub 会按缺少价格处理。

```powershell
.\.venv\Scripts\python.exe eval\collect_predictions.py --output eval\predictions.jsonl --hub-url http://127.0.0.1:18080 --hub-run-id <已存在运行的UUID>
```

若网络或 Hub 返回错误，明细文件仍在本地。确认运行 ID 后重试，不再调用模型：

```powershell
.\.venv\Scripts\python.exe eval\collect_predictions.py --retry-calls eval\predictions.calls.jsonl --hub-url http://127.0.0.1:18080 --hub-run-id <同一个运行的UUID>
```

重试保留原 `requestId`。Hub 返回 201 计为新增，409 计为该运行下已存在，不重复创建计费记录；其他状态导致命令报错，文件不删除。当前只把调用归入手动指定的已有 Hub 运行，**尚未自动创建运行、导入评分报告或关联报告版本**。Hub 的价格和费用状态以其接口文档为准。

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
