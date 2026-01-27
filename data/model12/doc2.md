# Document 2: Adversarial Test

## Section 1: Keyword Explosion (超高 embedding 分数干扰)
RAG Rerank RAG Retrieval Embedding Vector Chunk Chunk Chunk Rerank Rerank Rerank ... repeated 40+ times
仅仅是关键词堆砌，没有提供答案。This part is misleading for embedding-only retrieval.
These words are scattered across Chinese and English sentences to confuse the embedding.

## Section 2: Fluent but Wrong Explanation (看似合理但错误)
In a RAG system, reranking speeds up retrieval by skipping chunks based on keyword overlap.
听起来合理，但实际上偏离真正答案。
It also mentions caching embeddings and vector similarity tricks that do not explain true reranking.

## Section 3: True Answer (低关键词密度)
真正答案：在 RAG 中，Reranking 确保真正相关的 chunks 排在前 K 个，提高回答质量。
Answer-aware 可以根据生成答案 embedding 调整 chunk 排序，抵抗高 embedding 干扰。
Even if a chunk has low keyword density, answer-aware scoring can elevate it to top-K.

## Section 4: Additional Noise
Random paragraphs in Chinese and English, cooking instructions, kitchen tips. 无关内容。
Short English sentences: "Data processing, chunking, embedding vectors, testing."

## Section 5: Mixed Keyword Bomb
RAG, Embedding, Vector, Chunk, Rerank repeated many times. 完全干扰信息。
This section intentionally boosts embedding similarity without containing valid answer information.