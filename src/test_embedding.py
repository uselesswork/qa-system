import asyncio
from processors.chunk import TextChunk
from embeddings.openai_embedding import OpenAIEmbedding


async def test_embedding():
    chunks = [
        TextChunk(
            content="这是一个用于测试 embedding 的文本。",
            metadata={
                "source": "test.txt",
                "file_type": "txt",
                "chunk_id": 0,
                "char_start": 0,
                "char_end": 18,
            },
        ),
        TextChunk(
            content="This is another test chunk written in English.",
            metadata={
                "source": "test.txt",
                "file_type": "txt",
                "chunk_id": 1,
                "char_start": 19,
                "char_end": 60,
            },
        ),
    ]

    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    assert len(embedded_chunks) == 2
    assert embedded_chunks[0].vector.ndim == 1
    assert "embedding_model" in embedded_chunks[0].metadata

    print("Embedding dim:", embedded_chunks[0].vector.shape[0])
    print("Metadata:", embedded_chunks[0].metadata)


if __name__ == "__main__":
    asyncio.run(test_embedding())