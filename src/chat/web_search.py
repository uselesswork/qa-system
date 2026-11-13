"""显式联网问答；只把当前问题发送给搜索模型。"""

import os
from urllib.parse import urlsplit

from openai import AsyncOpenAI


class WebSearchUnavailable(Exception):
    pass


class WebSearchService:
    def __init__(self, client: AsyncOpenAI | None = None, model: str | None = None):
        self.client = client or AsyncOpenAI()
        self.model = model or os.getenv("WEB_SEARCH_MODEL", "gpt-6-astra")

    async def search(self, question: str) -> dict:
        response = await self.client.responses.create(
            model=self.model,
            tools=[{"type": "web_search", "search_context_size": "low"}],
            tool_choice="required",
            input=question,
            store=False,
        )
        answer = response.output_text or ""
        citations = []
        sources = []
        seen = set()
        for output in response.output:
            if output.type != "message":
                continue
            for content in output.content:
                if content.type != "output_text":
                    continue
                for annotation in content.annotations or []:
                    if annotation.type != "url_citation":
                        continue
                    url = annotation.url
                    parsed = urlsplit(url)
                    if parsed.scheme not in ("http", "https") or not parsed.netloc:
                        continue
                    title = annotation.title or parsed.netloc
                    start, end = annotation.start_index, annotation.end_index
                    citations.append({"url": url, "title": title,
                                      "start_index": start, "end_index": end})
                    if url not in seen:
                        seen.add(url)
                        sources.append({"source_type": "web", "title": title,
                                        "url": url, "rank": len(sources) + 1})
        if not answer.strip() or not sources:
            raise WebSearchUnavailable("联网搜索没有返回可核对的网页来源，请稍后重试。")
        return {"answer": answer, "evidence": sources,
                "metadata": {"mode": "web", "model": self.model,
                             "citations": citations}}
