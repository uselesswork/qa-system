from typing import List
from src.retrievers.base import BaseRetriever, RetrievalBatch
from src.vectorstores.base import BaseVectorStore
from src.vectorstores.search_result import SearchResult
from src.embeddings.base import BaseEmbedding


class SimpleRetriever(BaseRetriever):

    def __init__(
        self,
        embedder: BaseEmbedding,
        vectorstore: BaseVectorStore,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore

    async def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        return (await self.retrieve_with_vector(query=query, top_k=top_k)).results

    async def retrieve_with_vector(self, query: str, top_k: int = 5) -> RetrievalBatch:
        embedded = await self.embedder.embed_texts([query])
        query_vector = embedded[0].vector
        results = self.vectorstore.search(
            query_vector=query_vector,
            top_k=top_k
        )
        return RetrievalBatch(results=results, query_vector=query_vector)
