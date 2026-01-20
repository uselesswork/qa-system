import re
from typing import List
from src.processors.semantic.base import BaseSemanticSplitter
from src.processors.chunk import TextChunk
from src.loaders.document import Document


class HeadingSemanticSplitter(BaseSemanticSplitter):

    HEADING_PATTERN = re.compile(r"^(#{1,6})\\s+(.+)$", re.MULTILINE)

    def split(self, document: Document) -> List[TextChunk]:
        text = document.content
        matches = list(self.HEADING_PATTERN.finditer(text))

        if not matches:
            # 无标题，fallback
            return []

        chunks: List[TextChunk] = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue

            chunks.append(
                TextChunk(
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "semantic": "heading",
                        "heading": m.group(2),
                    }
                )
            )

        return  chunks
