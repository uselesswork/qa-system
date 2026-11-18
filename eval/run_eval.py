"""对固定样本和同一次运行导出的预测文件计算指标。"""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from metrics import evaluate


ROOT = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="评估同一次问答运行导出的预测 JSONL")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset.jsonl")
    parser.add_argument("--corpus", type=Path, help="本次预测使用的固定语料目录；记录每个 Markdown 文件哈希")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-label", required=True, help="模型和配置标识，例如 luna-20261116")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    dataset_bytes = args.dataset.read_bytes()
    predictions_bytes = args.predictions.read_bytes()
    report = evaluate(read_jsonl(args.dataset), read_jsonl(args.predictions), args.k)
    report["run"] = {
        "label": args.run_label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions_bytes).hexdigest(),
        "predictions_path": args.predictions.as_posix(),
    }
    if args.corpus is not None:
        files = sorted(args.corpus.glob("*.md"))
        if not files:
            raise ValueError("语料目录没有 Markdown 文件")
        report["run"]["corpus_sha256"] = {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已评估 {report['sample_count']} 条样本；报告：{args.output}")


if __name__ == "__main__":
    main()
