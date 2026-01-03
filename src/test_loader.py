from pathlib import Path
from loaders.txt_loader import TxtLoader
from loaders.md_loader import MarkdownLoader
from loaders.pdf_loader import PdfLoader

BASE_DIR = Path(__file__).resolve().parent.parent

def test_txt():
    loader = TxtLoader()
    docs = loader.load(BASE_DIR / "data" / "model1" /"test.txt")
    assert len(docs) == 1
    print(docs[0].content)

def test_md():
    loader = MarkdownLoader()
    docs = loader.load(BASE_DIR / "data" / "model1" /"test.md")
    print(docs[0].content)

def test_pdf():
    loader = PdfLoader()
    docs = loader.load(BASE_DIR / "data" / "model1" /"test.pdf")
    print(docs[0].content)

if __name__ == "__main__":
    test_txt()
    test_md()
    test_pdf()