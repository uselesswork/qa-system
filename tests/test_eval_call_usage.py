import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from eval.collect_predictions import CountingLLM, publish_if_enabled, upload_calls
from src.llms.openai_chat import OpenAIChatLLM


def test_call_records_actual_model_and_only_paired_supplier_usage():
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
    asyncio.run(llm.generate("private prompt"))
    asyncio.run(llm.generate("another private prompt"))

    assert llm.calls == 2
    assert len({row["requestId"] for row in records}) == 2
    assert all(row["caseId"] == "case-1" and row["modelId"] == "actual-luna"
               and row["latencyMs"] >= 0 for row in records)
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
                                "modelId": "actual-luna", "latencyMs": 12.5,
                                "inputTokens": 0, "outputTokens": 4}) + "\n", encoding="utf-8")
    posted = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"id": run_id})
        posted.append(json.loads(request.content))
        return httpx.Response(201 if len(posted) == 1 else 409)

    transport = httpx.MockTransport(handler)
    assert upload_calls(path, "http://127.0.0.1:18080", run_id, transport) == (1, 0)
    assert upload_calls(path, "http://127.0.0.1:18080", run_id, transport) == (0, 1)
    assert posted == [{"requestId": call_id, "modelId": "actual-luna",
                       "latencyMs": 12.5, "inputTokens": 0, "outputTokens": 4}] * 2
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
