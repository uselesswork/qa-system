from abc import ABC, abstractmethod
from typing import List
from src.processors.chunk import TextChunk
from src.loaders.document import Document


class BaseSemanticSplitter(ABC):

    @abstractmethod
    def split(self, document: Document) -> List[TextChunk]:
        """
        基于语义对 document 进行切分
        """
        pass
