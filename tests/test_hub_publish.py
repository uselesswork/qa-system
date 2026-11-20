import json
from uuid import uuid4

import httpx

from eval import publish_to_hub


def test_report_import_returns_run_id_for_call_upload(tmp_path, monkeypatch):
    report = tmp_path / "run.report.json"
    report.write_text('{"sample_count":1}', encoding="utf-8")
    calls = tmp_path / "run.calls.jsonl"
    calls.write_text('{"requestId":"call-1"}\n', encoding="utf-8")
    run_id = str(uuid4())
    seen = []

    def handler(request):
        seen.append((request.url.path, request.content))
        return httpx.Response(201, json={"runId": run_id, "imported": True})

    uploads = []
    monkeypatch.setattr(publish_to_hub, "upload_calls",
                        lambda path, url, run: uploads.append((path, url, run)) or (1, 0))
    assert publish_to_hub.publish(report, calls, "http://127.0.0.1:8080", "model-a",
                                  transport=httpx.MockTransport(handler)) == (run_id, 1, 0)
    assert seen[0][0] == "/api/imports/qa-system"
    assert uploads == [(calls, "http://127.0.0.1:8080", run_id)]
    assert json.loads(report.read_text(encoding="utf-8")) == {"sample_count": 1}
