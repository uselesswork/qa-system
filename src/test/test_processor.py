from pathlib import Path
from src.loaders.txt_loader import TxtLoader
from src.loaders.md_loader import MarkdownLoader
from src.loaders.pdf_loader import PdfLoader
from src.processors.cleaner import TextCleaner
from src.processors.splitter import TextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent.parent

cleaner = TextCleaner()
splitter = TextSplitter()


def txt_test():
    loader = TxtLoader()
    docs = loader.load(BASE_DIR / "data" / "basic_pipeline_examples" / "test.txt")

    for doc in docs:
        doc.content = cleaner.clean(doc.content)
        chunks = splitter.split(doc)

        print(f"Chunks: {len(chunks)}")
        for c in chunks:
            print("----")
            print(c.content)
            print(c.metadata)

def md_test():
    loader = MarkdownLoader()
    docs = loader.load(BASE_DIR / "data" / "basic_pipeline_examples" / "test.md")

    for doc in docs:
        doc.content = cleaner.clean(doc.content)
        chunks = splitter.split(doc)

        print(f"Chunks: {len(chunks)}")
        for c in chunks:
            print("----")
            print(c.content)
            print(c.metadata)

def pdf_test():
    loader = PdfLoader()
    docs = loader.load(BASE_DIR / "data" / "basic_pipeline_examples" / "test.pdf")

    for doc in docs:
        doc.content = cleaner.clean(doc.content)
        chunks = splitter.split(doc)

        print(f"Chunks: {len(chunks)}")
        for c in chunks:
            print("----")
            print(c.content)
            print(c.metadata)

if __name__ == "__main__":
    # txt_test()
    md_test()
    # pdf_test()
