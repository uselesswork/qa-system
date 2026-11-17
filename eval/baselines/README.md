# 2026-11-17 首次真实模型基线

- 采集命令：`.\.venv\Scripts\python.exe eval\collect_predictions.py --model gpt-6-luna`
- 评分命令：`.\.venv\Scripts\python.exe eval\run_eval.py --predictions eval\baselines\2026-11-17-gpt-6-luna.predictions.jsonl --output eval\baselines\2026-11-17-gpt-6-luna.report.json --run-label baseline-2026-11-17-gpt-6-luna`
- 配置：`gpt-6-luna`，默认服务端温度；`text-embedding-3-small`；`top_k=5`；Python 3.11.9。
- 语料 SHA-256：`noise.md` 为 `9ac94c2832213221425f3d26939d7e8053955d2ef65e92230524cde929ccfaa8`，`project.md` 为 `db1a8c8a6384fff76e390567a573e37af88d25086118e214e851eae732ea120c`，`team.md` 为 `ad1b2dff0ebcde63b9c28ce7d95e8674010679a037c77ea3f8c04ac67e2f2091`。数据集及预测哈希见报告。
- 结果：5 条中 4 条可回答题的 `Hit@5`、`MRR@5`、引用准确率、答案包含率均为 `1.0`；1 条无答案题正确拒答。P50 `4659.872 ms`，P95 `6710.773 ms`，共 10 次 LLM 调用。

这份基线使用现有 RAG 组件和真实 Embedding/LLM，向量库为内存实现。它只适合对同一小语料做回归比较，不能代表 MySQL/Qdrant/浏览器全链路的质量或延迟；费用未记录。
