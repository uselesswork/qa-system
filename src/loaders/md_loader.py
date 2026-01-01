from .base import BaseDocumentLoader
from .document import Document

class MarkdownLoader(BaseDocumentLoader):

    def load(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        return [
            Document(
                content=text,
                metadata={
                    "source": file_path,
                    "file_type": "md"
                }
            )
        ]
