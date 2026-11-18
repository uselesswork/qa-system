import pdfplumber
from .base import BaseDocumentLoader
from .document import Document

class PdfLoader(BaseDocumentLoader):

    def load(self, file_path: str):
        documents = []
        extracted_chars = 0

        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) > 120:
                raise ValueError("PDF 页数超过 120 页。")
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                extracted_chars += len(text)
                if extracted_chars > 500_000:
                    raise ValueError("PDF 提取文字超过 500000 字符。")

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
