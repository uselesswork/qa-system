# 阶段 14：固定样本评测

`dataset.jsonl` 是五条人工标注的小样本，资料在 `corpus/`。`relevant_sources` 以原始文件名为单位标注，可同时有多个正确来源。样本覆盖直接问答、跨文档、指代追问、无答案和噪声干扰。它只能做回归观察，不代表生产数据分布。

可选择在独立语料上运行真实模型采集预测。此命令会调用 Embedding 和 LLM，产生 API 费用；默认 LLM 是 `gpt-6-luna`，不会使用 Astra：

```powershell
.\.venv\Scripts\python.exe eval\collect_predictions.py --model gpt-6-luna
```

评测脚本读取**同一次系统运行**产生的 `predictions.jsonl`，每条记录格式：

```json
{"id":"direct","retrieved_sources":["project.md"],"cited_sources":["project.md"],"answer":"2026年6月18日","refused":false,"latency_ms":120.0,"model_calls":2}
```

`retrieved_sources` 按召回排名填写，即使最终没有引用也要记录；`cited_sources` 填最终展示给用户的来源。无答案时 `refused=true`。多轮样本需按 `history` 先建立上下文，记录最后一问的预测。每个 `id` 必须有一条预测；漏项或重复项直接报错。已有页面接口只返回最终引用，不返回完整召回排名，因此不能只把页面历史当作完整预测文件；要在评测运行中另行采集召回顺序。

```powershell
.\.venv\Scripts\python.exe eval\run_eval.py --predictions eval\predictions.jsonl --output eval\reports\run.json --run-label luna-20261116
```

报告记录样本数、`k`、运行标签、UTC 时间、Python 版本、输入文件 SHA-256，以及逐条结果。`Hit@k` 和 `MRR@k` 仅统计有标准证据的样本；引用准确率是每条可回答样本的“正确引用数 ÷ 引用总数”的平均值；无答案正确率只统计无答案样本。答案包含率是简单字符串核对，不等于语义正确率。报告中的延迟和模型调用次数完全取自预测文件，脚本不会伪造它们。

当前仅运行了无密钥回归，尚未执行付费采集，也没有有效的模型质量基线。需要先固定模型、参数和语料版本，在同一套样本上采集改造前后预测，再比较指标。采集脚本直接运行现有 RAG 组件与独立语料，不涉及 MySQL、Redis 或 Docker。
