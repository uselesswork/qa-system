from src.prompts.simple_prompt import SimplePromptBuilder
from src.vectorstores.search_result import SearchResult
from src.embeddings.embedded_chunk import EmbeddedChunk
import numpy as np


def test_prompt_builder():
    query = "什么是 RAG 系统？"

    # 构造假的 EmbeddedChunk（向量在这里不重要）
    chunk1 = EmbeddedChunk(
        content="RAG 是一种结合检索与生成的技术。",
        vector=np.zeros(3),
        metadata={}
    )

    chunk2 = EmbeddedChunk(
        content="RAG 系统通过向量检索增强大模型回答能力。",
        vector=np.zeros(3),
        metadata={}
    )

    # 构造 SearchResult
    results = [
        SearchResult(chunk=chunk1, score=0.9),
        SearchResult(chunk=chunk2, score=0.8),
    ]

    builder = SimplePromptBuilder()
    prompt = builder.build(query=query, chunks=results)

    print(prompt)


if __name__ == "__main__":
    test_prompt_builder()
