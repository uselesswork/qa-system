import sys
import pdfplumber

def main():
    print("Python version:", sys.version)
    print("pdfplumber version:", pdfplumber.__version__)

    # 简单文本测试
    with open("../../data/test.txt", "r", encoding="utf-8") as f:
        content = f.read()
        print(content)

    assert len(content) > 0
    print("Text file read successfully ✅")

    print("Environment setup SUCCESS 🎉")

if __name__ == "__main__":
    main()
