# Document 1: Adversarial Test

## Section 1: Keyword Bomb (大量干扰)
RAG RAG RAG Retrieval Embedding Vector Chunk Chunk Chunk Reranking Rerank Rerank Rerank.
These keywords are repeated many times. 这段只是堆砌关键词，embedding 分数高，但没有有效信息。
Nothing here explains reranking.
Even more repeated keywords: RAG, Rerank, Chunk, Embedding, Vector. 中文描述和英文关键词混杂，完全无关。

## Section 2: Misleading Explanation (看似合理但错误)
In a RAG system, reranking is mainly used to speed up retrieval by skipping chunks.
这段听起来合理，但实际上偏离真正答案。
The explanation mentions caching vectors, keyword overlap, and skipping chunks, but gives no real insight.

## Section 3: True Answer (低关键词密度)
真正答案：在 RAG 系统中，Reranking 会在初始向量召回后，根据 query 重新排序 chunks。
Answer-aware aggregator 会根据生成答案 embedding 对 chunks 再排序，即使 embedding 被干扰，也能识别真正答案。
This section provides the correct reasoning behind Reranking: semantic relevance matters, not keyword density.

## Section 4: Extra Noise
Cooking, recipes, kitchen tools, general instructions. 与 RAG 无关。
Random unrelated English sentences mixed with Chinese phrases.

## Section 5: Mixed Noise
RAG, Rerank, Chunk, Embedding, Vector repeated multiple times. 完全无效信息。
Short phrases, lists, random numbers: 12345, test, demo, chunking, retrieval.