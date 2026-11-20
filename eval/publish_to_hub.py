"""将现有评分报告和配对调用明细上报到已有的本机 Hub。"""

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.collect_predictions import upload_calls  # noqa: E402


def publish(report: Path, calls: Path, hub_url: str, model_id: str,
            result_source: str = "LIVE_MODEL",
            transport: httpx.BaseTransport | None = None) -> tuple[str, int, int]:
    if not report.is_file() or not calls.is_file():
        raise FileNotFoundError("报告或调用明细不存在；不会重新运行模型。")
    url = f"{hub_url.rstrip('/')}/api/imports/qa-system"
    with httpx.Client(transport=transport, trust_env=False, timeout=10) as client:
        with report.open("rb") as handle:
            response = client.post(url, files={"report": (report.name, handle, "application/json")},
                                   data={"resultSource": result_source, "modelId": model_id})
    if response.status_code not in (200, 201):
        raise RuntimeError(f"Hub 报告导入返回 HTTP {response.status_code}；本地文件已保留。")
    run_id = response.json()["runId"]
    created, duplicates = upload_calls(calls, hub_url, run_id)
    return run_id, created, duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description="只上报已有评分报告和调用明细，不调用模型")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--calls-file", type=Path, required=True)
    parser.add_argument("--hub-url", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--result-source", choices=("LIVE_MODEL", "OFFLINE_FIXTURE"), default="LIVE_MODEL")
    args = parser.parse_args()
    run_id, created, duplicates = publish(args.report, args.calls_file, args.hub_url,
                                          args.model_id, args.result_source)
    print(f"Hub 运行 {run_id}：调用新增 {created} 条，已存在 {duplicates} 条。")


if __name__ == "__main__":
    main()
