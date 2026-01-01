import pdfplumber
from .base import BaseDocumentLoader
from .document import Document

class PdfLoader(BaseDocumentLoader):

    def load(self, file_path: str):
        documents = []

        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""

                documents.append(
                    Document(
                        content=text,
                        metadata={
                            "source": file_path,
                            "file_type": "pdf",
                            "page": i + 1
                        }
                    )
                )

        return documents