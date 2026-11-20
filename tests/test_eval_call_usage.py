import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from eval.collect_predictions import CountingLLM, collect, main, publish_if_enabled, upload_calls
from src.llms.openai_chat import OpenAIChatLLM


def test_call_records_actual_model_integer_latency_and_only_paired_usage(monkeypatch):
    records = []
    answers = iter([
        SimpleNamespace(metadata={"model": "actual-luna", "prompt_tokens": 11,
                                  "completion_tokens": 7}, content="answer"),
        SimpleNamespace(metadata={"model": "actual-luna", "prompt_tokens": 9},
                        content="answer"),
    ])

    class FakeLLM:
        async def generate(self, _prompt):
            return next(answers)

    llm = CountingLLM("requested-luna", records.append, inner=FakeLLM())
    llm.case_id = "case-1"
    ticks = iter([10.0, 10.0126, 20.0, 20.0004])
    monkeypatch.setattr("eval.collect_predictions.time.perf_counter", lambda: next(ticks))
    asyncio.run(llm.generate("private prompt"))
    asyncio.run(llm.generate("another private prompt"))

    assert llm.calls == 2
    assert len({row["requestId"] for row in records}) == 2
    assert all(row["caseId"] == "case-1" and row["modelId"] == "actual-luna"
               for row in records)
    assert [row["latencyMs"] for row in records] == [13, 0]
    assert all(type(row["latencyMs"]) is int for row in records)
    assert records[0]["inputTokens"] == 11
    assert records[0]["outputTokens"] == 7
    assert "inputTokens" not in records[1] and "outputTokens" not in records[1]
    assert all("prompt" not in json.dumps(row) and "answer" not in json.dumps(row)
               for row in records)


def test_openai_response_model_and_missing_usage_are_preserved():
    async def create(**_kwargs):
        return SimpleNamespace(model="actual-model", usage=None,
                               choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    answer = asyncio.run(OpenAIChatLLM(model="requested-model", client=client).generate("q"))
    assert answer.metadata["model"] == "actual-model"
    assert "prompt_tokens" not in answer.metadata
    assert "completion_tokens" not in answer.metadata


def test_hub_upload_retry_keeps_id_and_handles_409(tmp_path):
    run_id = str(uuid4())
    call_id = str(uuid4())
    path = tmp_path / "predictions.calls.jsonl"
    path.write_text(json.dumps({"requestId": call_id, "caseId": "private-case",
                                "modelId": "actual-luna", "latencyMs": 13,
                                "inputTokens": 0, "outputTokens": 4}) + "\n", encoding="utf-8")
    posted = []

    def handler(request):
        if request.method == "GET":
            if request.url.path.endswith("/usage"):
                return httpx.Response(200, json={"calls": [
                    {"requestId": call_id, "modelId": "actual-luna", "latencyMs": 13,
                     "inputTokens": 0, "outputTokens": 4, "priceVersion": None}]})
            return httpx.Response(200, json={"id": run_id})
        posted.append(json.loads(request.content))
        return httpx.Response(201 if len(posted) == 1 else 409)

    transport = httpx.MockTransport(handler)
    assert upload_calls(path, "http://127.0.0.1:18080", run_id, transport) == (1, 0)
    assert upload_calls(path, "http://127.0.0.1:18080", run_id, transport) == (0, 1)
    assert posted == [{"requestId": call_id, "modelId": "actual-luna",
                       "latencyMs": 13, "inputTokens": 0, "outputTokens": 4}] * 2
    assert call_id in path.read_text(encoding="utf-8")


def test_missing_usage_failure_and_default_off(tmp_path, monkeypatch):
    path = tmp_path / "calls.jsonl"
    path.write_text(json.dumps({"requestId": str(uuid4()), "caseId": "c",
                                "modelId": "m", "latencyMs": 5,
                                "inputTokens": 8}) + "\n", encoding="utf-8")
    monkeypatch.setattr("eval.collect_predictions.upload_calls",
                        lambda *_args, **_kwargs: pytest.fail("default must not contact Hub"))
    assert publish_if_enabled(path, None, None) is None
    monkeypatch.undo()
    run_id = str(uuid4())
    posted = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200)
        posted.append(json.loads(request.content))
        return httpx.Response(503)

    with pytest.raises(RuntimeError, match="--retry-calls"):
        upload_calls(path, "http://127.0.0.1:18080", run_id,
                     httpx.MockTransport(handler))
    assert "inputTokens" not in posted[0] and "outputTokens" not in posted[0]
    assert path.exists()


@pytest.mark.parametrize("remote", [
    {"calls": []},
    {"calls": [{"requestId": "same", "modelId": "other", "latencyMs": 2,
                "inputTokens": None, "outputTokens": None, "priceVersion": None}]},
    {"calls": [{"requestId": "same", "modelId": "m", "latencyMs": 2,
                "inputTokens": 1, "outputTokens": 1, "priceVersion": None}]},
    {"calls": [{"requestId": "same", "modelId": "m", "latencyMs": 3,
                "inputTokens": None, "outputTokens": None, "priceVersion": None}]},
    {"calls": [{"requestId": "same", "modelId": "m", "latencyMs": 2,
                "inputTokens": None, "outputTokens": None, "priceVersion": "other"}]},
])
def test_409_missing_or_conflicting_record_is_not_success(tmp_path, remote):
    path = tmp_path / "calls.jsonl"
    path.write_text(json.dumps({"requestId": "same", "modelId": "m", "latencyMs": 2}) + "\n")

    def handler(request):
        if request.url.path.endswith("/usage"):
            return httpx.Response(200, json=remote)
        return httpx.Response(200 if request.method == "GET" else 409)

    with pytest.raises(RuntimeError, match="409"):
        upload_calls(path, "http://127.0.0.1:18080", str(uuid4()),
                     httpx.MockTransport(handler))
    assert path.exists()


def test_409_usage_lookup_failure_is_not_success(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text(json.dumps({"requestId": "same", "modelId": "m", "latencyMs": 2}) + "\n")

    def handler(request):
        if request.url.path.endswith("/usage"):
            return httpx.Response(503)
        return httpx.Response(200 if request.method == "GET" else 409)

    with pytest.raises(RuntimeError, match="查询返回 HTTP 503"):
        upload_calls(path, "http://127.0.0.1:18080", str(uuid4()),
                     httpx.MockTransport(handler))


def test_existing_calls_file_blocks_collection_before_model_or_env(tmp_path, monkeypatch):
    path = tmp_path / "predictions.calls.jsonl"
    path.write_text('{"requestId":"keep-me"}\n', encoding="utf-8")
    monkeypatch.setattr("eval.collect_predictions.load_dotenv",
                        lambda *_args: pytest.fail("must stop before environment access"))
    with pytest.raises(FileExistsError, match="--retry-calls"):
        asyncio.run(collect(tmp_path / "dataset.jsonl", tmp_path / "predictions.jsonl",
                            "fake-model", calls_output=path))
    assert path.read_text(encoding="utf-8") == '{"requestId":"keep-me"}\n'


def test_retry_cli_only_reuses_file_and_does_not_collect(tmp_path, monkeypatch):
    path = tmp_path / "calls.jsonl"
    path.write_text('{"requestId":"keep-me"}\n', encoding="utf-8")
    run_id = str(uuid4())
    monkeypatch.setattr("eval.collect_predictions.collect",
                        lambda *_args: pytest.fail("retry must not call model"))
    seen = []
    monkeypatch.setattr("eval.collect_predictions.upload_calls",
                        lambda *args: seen.append(args) or (0, 1))
    monkeypatch.setattr("sys.argv", ["collect_predictions.py", "--retry-calls", str(path),
                                 "--hub-url", "http://127.0.0.1:18080", "--hub-run-id", run_id])
    main()
    assert seen == [(path, "http://127.0.0.1:18080", run_id)]
