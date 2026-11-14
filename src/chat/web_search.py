"""显式联网问答；只把当前问题发送给搜索模型。"""

import os
from urllib.parse import urlsplit

from openai import AsyncOpenAI


class WebSearchUnavailable(Exception):
    pass


class WebSearchService:
    def __init__(self, client: AsyncOpenAI | None = None, model: str | None = None):
        self.client = client or AsyncOpenAI()
        self.model = model or os.getenv("WEB_SEARCH_MODEL", "gpt-6-luna")

    async def search(self, question: str) -> dict:
        response = await self.client.responses.create(
            model=self.model,
            tools=[{"type": "web_search", "search_context_size": "low"}],
            tool_choice="required",
            include=["web_search_call.action.sources"],
            input=question,
            store=False,
        )
        answer = response.output_text or ""
        citations = []
        sources = []
        seen = set()
        for output in response.output:
            if output.type == "web_search_call":
                for source in getattr(getattr(output, "action", None), "sources", None) or []:
                    url = getattr(source, "url", None)
                    if not isinstance(url, str):
                        continue
                    parsed = urlsplit(url)
                    if parsed.scheme not in ("http", "https") or not parsed.netloc or url in seen:
                        continue
                    seen.add(url)
                    sources.append({"source_type": "web", "title": getattr(source, "title", None) or parsed.netloc,
                                    "url": url, "rank": len(sources) + 1})
                continue
            if output.type != "message":
                continue
            for content in output.content:
                if content.type != "output_text":
                    continue
                if not answer:
                    answer = getattr(content, "text", "") or ""
                for annotation in content.annotations or []:
                    if annotation.type != "url_citation":
                        continue
                    url = getattr(annotation, "url", None)
                    if not isinstance(url, str):
                        continue
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
        if not answer.strip():
            raise WebSearchUnavailable("联网搜索没有返回文字答案，请稍后重试。")
        if not sources:
            raise WebSearchUnavailable("联网搜索没有返回可核对的网页来源，请稍后重试。")
        if not citations:
            for source in sources[:3]:
                marker = f"【{source['rank']}】"
                answer += f"\n来源：{marker}"
                citations.append({"url": source["url"], "title": source["title"],
                                  "start_index": len(answer) - len(marker),
                                  "end_index": len(answer)})
        return {"answer": answer, "evidence": sources,
                "metadata": {"mode": "web", "model": self.model,
                             "citations": citations}}
