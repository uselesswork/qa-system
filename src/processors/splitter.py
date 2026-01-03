from typing import List
from .chunk import TextChunk
from ..loaders.document import Document


class TextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> List[TextChunk]:
        text = document.content
        if not text or not text.strip():
            return []

        chunks: List[TextChunk] = []
        text_len = len(text)

        start = 0
        chunk_id = 0

        while start < text_len:
            # 1. 先给一个“硬上限 end”
            end = min(start + self.chunk_size, text_len)

            # 2. 尝试在句子边界回退（只回退，不前进）
            window = text[start:end]
            for sep in ["。", "！", "？", ".", "!", "?"]:
                pos = window.rfind(sep)
                if pos != -1 and pos > len(window) * 0.6:
                    end = start + pos + 1
                    break

            # 3. 兜底：保证 end 一定前进
            if end <= start:
                end = min(start + self.chunk_size, text_len)

            chunk_text = text[start:end]

            chunks.append(
                TextChunk(
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_id": chunk_id,
                        "char_start": start,
                        "char_end": end,
                    },
                )
            )

            chunk_id += 1

            # 4. 计算下一轮 start，确保单调递增
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = end

            start = next_start

        return chunks
