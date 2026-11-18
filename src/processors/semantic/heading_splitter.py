import re
from typing import List
from src.processors.semantic.base import BaseSemanticSplitter
from src.processors.chunk import TextChunk
from src.loaders.document import Document
from src.processors.splitter import TextSplitter


class HeadingSemanticSplitter(BaseSemanticSplitter):

    HEADING_PATTERN = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)

    def split(self, document: Document) -> List[TextChunk]:
        text = document.content
        matches = list(self.HEADING_PATTERN.finditer(text))

        if not matches:
            # 无标题，fallback
            return []

        chunks: List[TextChunk] = []

        def add_chunk(start: int, end: int, heading: str | None):
            raw = text[start:end]
            content = raw.strip()
            if not content:
                return
            char_start = start + len(raw) - len(raw.lstrip())
            char_end = end - (len(raw) - len(raw.rstrip()))
            metadata = {**document.metadata, "semantic": "heading", "heading": heading}
            if len(content) <= 500:
                chunks.append(TextChunk(content=content, metadata={
                    **metadata, "chunk_id": len(chunks),
                    "char_start": char_start, "char_end": char_end,
                }))
                return
            for part in TextSplitter().split(Document(content=content, metadata=metadata)):
                part.metadata["char_start"] += char_start
                part.metadata["char_end"] += char_start
                part.metadata["chunk_id"] = len(chunks)
                chunks.append(part)

        # 标题前的说明文字也属于文档，不能在标题切分时丢弃。
        add_chunk(0, matches[0].start(), None)
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            add_chunk(match.start(), end, match.group(2).strip())

        return chunks
