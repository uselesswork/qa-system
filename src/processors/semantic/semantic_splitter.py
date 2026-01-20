from typing import List
from src.loaders.document import Document
from src.processors.chunk import TextChunk
from src.processors.splitter import TextSplitter
from src.processors.semantic.heading_splitter import HeadingSemanticSplitter


class SemanticSplitter:


    def __init__(self):
        self.heading_splitter = HeadingSemanticSplitter()
        self.fallback_splitter = TextSplitter()


    def split(self, document: Document) -> List[TextChunk]:
        if document.metadata.get("file_type") == "md":
            chunks = self.heading_splitter.split(document)

            if chunks:
                return chunks

        return self.fallback_splitter.split(document)
