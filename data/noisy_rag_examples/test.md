# RAG 系统对抗测试文档

## Section 1: Noise / 噪声段落
RAG, Retrieval, Embedding, Vector, Chunk, Reranking. RAG, RAG, RAG, Rerank, Rerank, Rerank. 
This paragraph is full of keywords but does not explain anything about Reranking. 
它只是重复关键词，没有实际信息。

## Section 2: Partial English / 半信息段
RAG is a system combining retrieval and generation. Sometimes it uses embeddings for chunk retrieval.
但是这里并没有说明 Reranking 的具体作用，只提到 retrieval 和 embeddings。

## Section 3: True Answer / 真正答案段
在 RAG 系统中，Reranking 的作用是在初始向量召回之后，根据 query 与每个 chunk 的语义相关性对候选结果重新排序。 
这确保了真正回答用户问题的内容排在最前面，从而提升 top-K 结果的准确性。
In other words, Reranker focuses on relevance, not just embedding similarity.

## Section 4: More Noise / 更多噪声
Cooking recipes, kitchen tools, and food preparation. 这些内容与 RAG 系统完全无关。
RAG, chunk, embedding, vector, retrieval, Rerank, Rerank. 只是关键词重复。

## Section 5: Semi-relevant
Retrieval-Augmented Generation (RAG) sometimes combines embeddings and LLMs to generate answers. 
但这里仍没有涉及 Reranking 的核心作用，只是背景知识。

## Section 6: Summary / 答案集中段
最终，在 RAG 系统中，Reranking 的关键是 **提升语义相关性排序**。 
Without reranking, top chunks may be keyword-heavy but not informative. 
这段会被 Reranker 排到前面。

## Section 7: English explanation
Reranking is the second stage after retrieval. 
It improves precision by putting the most relevant chunks first. 
This helps answer the question correctly when there is noisy retrieval.

## Section 8: Noise mix
在 RAG 系统中，chunk size 是一个非常关键的参数。 
如果 chunk 太小，语义不完整；chunk 太大，又会引入噪声。 
These sentences are about chunk size, not Reranking.