import asyncio
import json

import httpx

from app.main import app


def test_request_id_and_structured_access_log(caplog):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/")

    with caplog.at_level("INFO", logger="qa.http"):
        response = asyncio.run(run())
    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    event = json.loads(next(item.message for item in caplog.records
                            if item.name == "qa.http" and 'http_request' in item.message))
    assert event["request_id"] == request_id
    assert event["method"] == "GET" and event["path"] == "/"
    assert event["status"] == 200 and event["latency_ms"] >= 0
