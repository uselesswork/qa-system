# Document 3: Adversarial Test

## Section 1: Noise Mix (混合干扰)
RAG, Retrieval, Embedding, Vector, Chunk, Rerank. 中文描述与 English keywords 混杂。完全无关，embedding 相似度被提升。
Additional phrases: "top-K, semantic similarity, chunk splitting, irrelevant content."

## Section 2: True Answer (低关键词密度)
真正答案：Reranking 排序依据语义相关性，而不仅仅是关键词。
Answer-aware ranking 会根据最终生成的答案 embedding 对 chunks 重新排序，确保真正答案排在前 K。
This section describes the correct usage of semantic ranking and answer-aware aggregation.

## Section 3: High Noise
Cooking instructions, kitchen tips, unrelated text. 与 RAG 无关。
Random numbers and letters: ABC123, xyz456, 7890, demo.

## Section 4: Misleading Explanation
Reranking is mainly used to reduce computation time in retrieval. 看似合理但偏离真正答案。
Embedding similarity may rank this section high, misleading embedding-only retrieval.

## Section 5: Mixed Noise
RAG RAG Rerank Embedding Vector Chunk repeated many times. 完全无效信息。
This section contains random phrases in English and Chinese to boost fake relevance.