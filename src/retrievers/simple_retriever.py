from typing import List
from src.retrievers.base import BaseRetriever
from src.vectorstores.base import VectorStore
from src.vectorstores.search_result import SearchResult
from src.embeddings.base import BaseEmbedding


class SimpleRetriever(BaseRetriever):

    def __init__(
        self,
        embedder: BaseEmbedding,
        vectorstore: VectorStore,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore

    async def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:

        # 1. query → embedding
        embedded = await self.embedder.embed_texts([query])
        query_vector = embedded[0].vector

        # 2. vector search
        results = self.vectorstore.search(
            query_vector=query_vector,
            top_k=top_k
        )

        return results
