from abc import ABC, abstractmethod
from typing import List
from .document import Document

class BaseDocumentLoader(ABC):

    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """
        读取文件并返回 Document 列表
        """
        pass