# 2026-11-17 首次真实模型基线

- 采集命令：`.\.venv\Scripts\python.exe eval\collect_predictions.py --model gpt-6-luna`
- 评分命令：`.\.venv\Scripts\python.exe eval\run_eval.py --predictions eval\baselines\2026-11-17-gpt-6-luna.predictions.jsonl --output eval\baselines\2026-11-17-gpt-6-luna.report.json --run-label baseline-2026-11-17-gpt-6-luna`
- 配置：`gpt-6-luna`，默认服务端温度；`text-embedding-3-small`；`top_k=5`；Python 3.11.9。
- 语料 SHA-256：`noise.md` 为 `9ac94c2832213221425f3d26939d7e8053955d2ef65e92230524cde929ccfaa8`，`project.md` 为 `db1a8c8a6384fff76e390567a573e37af88d25086118e214e851eae732ea120c`，`team.md` 为 `ad1b2dff0ebcde63b9c28ce7d95e8674010679a037c77ea3f8c04ac67e2f2091`。数据集及预测哈希见报告。
- 结果：5 条中 4 条可回答题的 `Hit@5`、`MRR@5`、引用准确率、答案包含率均为 `1.0`；1 条无答案题正确拒答。P50 `4659.872 ms`，P95 `6710.773 ms`，共 10 次 LLM 调用。

这份基线使用现有 RAG 组件和真实 Embedding/LLM，向量库为内存实现。它只适合对同一小语料做回归比较，不能代表 MySQL/Qdrant/浏览器全链路的质量或延迟；费用未记录。

## 2026-11-18 阶段 16 复跑与扩展

- [`2026-11-18-stage14-regression-luna.report.json`](2026-11-18-stage14-regression-luna.report.json)：原五题、原语料、内存向量、同模型复跑；质量指标和 10 次 LLM 调用保持不变，延迟 P50/P95 为 7367/10142 ms。与上一日延迟差异不能单凭两次运行归因。
- [`2026-11-18-stage16-qdrant-luna.report.json`](2026-11-18-stage16-qdrant-luna.report.json)：独立七份语料、十题、临时本地 Qdrant；8 条可回答题命中且引用正确，2 条无答案题拒答，P50/P95 为 4957/6816 ms，共 19 次 LLM 调用。对应 `.predictions.jsonl` 保留逐题答案与召回来源，报告保存数据集、预测和逐文件语料哈希。

新十题和旧五题不是同一数据集，不能把指标差值当成检索优化效果。详见[阶段 16 文档](../../docs/第十六阶段-检索质量与安全加固.md)。
