# **模块0：环境搭建**

## **项目结构**

![model_0_1](md_img/model_0_1.png)



## 添加依赖（requirements.txt）：

```
# -----------------------------
# 模块 1：Document Loader 依赖
# -----------------------------
pdfplumber>=0.10.0
```



## 测试数据（test.txt）

```
这是一个测试文档。
用于验证问答系统的基础环境是否搭建成功。
```



## **控制台运行指令**

```
pip install -r requirements.txt
```



## 验收测试（**test_env.py**）

```
import sys
import pdfplumber

def main():
    print("Python version:", sys.version)
    print("pdfplumber version:", pdfplumber.__version__)

    # 简单文本测试
    with open("../data/test.txt", "r", encoding="utf-8") as f:
        content = f.read()
        print(content)

    assert len(content) > 0
    print("Text file read successfully ✅")

    print("Environment setup SUCCESS 🎉")

if __name__ == "__main__":
    main()
```

 

## 测试结果

![model_0_2](md_img/model_0_2.png)

 

# 模块1：文档加载器（Document Loader）



## **模块定位**

把各种格式的原始文档 → 转换成统一的 Document 结构



**明确不承担的职责（What NOT）**

-  不做文本清洗
-  不做切分
-  不关心 chunk
-  不关心语义
-  不关心向量 / 检索 / 模型

 

## **输入**

```
txt，pdf，md
```

 

## **输出**

```
Document {
    content: str
    metadata: dict
}

metadata: {
    "source": "文件路径",
    "file_type": "txt / md / pdf",
    "page": 1      # PDF 才有
}
```

 

## **项目结构**

```
src/
├─ loaders/
    ├─ __init__.py
    ├─ base.py
    ├─ document.py
    ├─ txt_loader.py
    ├─ md_loader.py
    └─ pdf_loader.py
```



![model_1_1](md_img/model_1_1.png)



## **核心数据结构**

```
src/loaders/document.py
```



```
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
```

**说明：**所有后续模块**都只认这个结构**



## Loader 抽象接口

### **为什么要抽象？**

后面可以：换解析库，加新格式

```
src/loaders/base.py
```



```
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
```



## 具体 Loader 实现

### TXT Loader

```
src/loaders/txt_loader.py
```



```
from .base import BaseDocumentLoader
from .document import Document

class TxtLoader(BaseDocumentLoader):

    def load(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        return [
            Document(
                content=text,
                metadata={
                    "source": file_path,
                    "file_type": "txt"
                }
            )
        ]
```



### Markdown Loader

```
src/loaders/md_loader.py
```



```
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
```



### PDF Loader

```
src/loaders/pdf_loader.py
```



```
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
```



## 模块 1 验收测试

```
src/test_loader.py
```



```
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
```



## 测试结果

![model_1_2](md_img/model_1_2.png)



# 模块2：文本处理器（Text Processor）

默认策略 = 中英混合优先



## 模块定位

把 Document → 切成“可向量化、可检索、可溯源”的 Chunk



**切分共识**

- 中英混合
- 句末优先
- 长句兜底
- chunk_size ≈ 500
- overlap ≈ 50–80
- 永不死循环



**明确不承担的职责**

- 不做 embedding
- 不做相似度
- 不做召回
- 不优化“检索效果”
- 不为模型服务



## 输入

```
List[Document]
```



## 输出

```
TextChunk {
    content: str
    metadata: dict
}

metadata {
	"source": str,        # 文件路径或唯一标识
	"file_type": str,     # txt / md / pdf
	"page": int | None,   # PDF 页码
	"chunk_id": int,      # 文档内 chunk 编号
	"char_start": int,    # 起始字符位置
	"char_end": int       # 结束字符位置
}
```



## Chunk 数据结构设计（核心）

### **为什么不能直接用 Document？**

- 一个 Document 可能非常长
- RAG 检索粒度必须更细
- 需要记录“来源 + 位置”



## 项目结构

```
src/
├─ processors/
    ├─ __init__.py
    ├─ cleaner.py
    ├─ splitter.py
    └─ chunk.py
```



![model_2_1](md_img/model_2_1.png)



## TextChunk 设计

```
src/processors/chunk.py
```



```
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TextChunk:
    """
    TextChunk 是 Text Processor 模块的最终输出单位

    metadata 字段规范：
    {
        "source": str,        # 文件路径或唯一标识
        "file_type": str,     # txt / md / pdf
        "page": int | None,   # PDF 页码
        "chunk_id": int,      # 文档内 chunk 编号
        "char_start": int,    # 起始字符位置
        "char_end": int       # 结束字符位置
    }
    """
    content: str
    metadata: Dict[str, Any]
```



## 文本清洗器（TextCleaner）

### **目标**

- 去掉无意义字符
- 保留语义结构
- 不“过度清洗”

```
src/processors/cleaner.py
```



```
import re

class TextCleaner:

    def clean(self, text: str) -> str:
        if not text:
            return ""

        # 去除多余空白
        text = re.sub(r"\s+", " ", text)

        # 去除控制字符
        text = re.sub(r"[\x00-\x1f\x7f]", "", text)

        return text.strip()
```

### **注意**

- 不做分词
- 不 lowercase
- 不去标点（对中文很重要）



## 中英混合切分器（Sentence Splitter）

### 设计目标

- 语言无关
- 结构优先
- 长度兜底



### 切分策略

- 段落
- 中英句子（正则）
- 字符长度兜底
- overlap 保留上下文



### 算法流程

```
1. 初始化检查
检查 chunk_overlap < chunk_size，否则抛出异常
处理空文本或空白文本，直接返回空列表
2. 主循环切割（每次循环生成一个chunk）
while 起始位置 < 文本长度:
    # 第一步：计算基准结束位置
    结束位置 = min(起始位置 + chunk_size, 文本长度)
    
    # 第二步：语义边界回退
    当前窗口文本 = 文本[起始位置:结束位置]
    for 每个句子分隔符 in ["。", "！", "？", ".", "!", "?"]:
        在窗口中从后往前找分隔符
        如果找到 且 位置在窗口后40%:
            将结束位置调整到分隔符后
            跳出循环
    
    # 第三步：防回退保护
    如果 结束位置 ≤ 起始位置:
        结束位置 = min(起始位置 + chunk_size, 文本长度)
    
    # 第四步：创建chunk
    创建TextChunk对象，包含：
        - 切片文本
        - 元数据（文档元数据 + chunk_id + 字符起止位置）
    
    # 第五步：防重复检查
    如果 当前chunk结束位置 ≤ 前一个chunk结束位置:
        删除当前chunk并跳出循环
    
    # 第六步：更新起始位置
    下一块起始位置 = 结束位置 - chunk_overlap
    如果 下一块起始位置 ≤ 当前起始位置:
        下一块起始位置 = 结束位置
```



```
src/processors/splitter.py
```



```
from typing import List
from .chunk import TextChunk
from ..loaders.document import Document


class TextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> List[TextChunk]:
        text = document.content
        if not text or not text.strip():
            return []

        chunks: List[TextChunk] = []
        text_len = len(text)

        start = 0
        chunk_id = 0

        while start < text_len:
            # 1. 先给一个“硬上限 end”
            end = min(start + self.chunk_size, text_len)

            # 2. 尝试在句子边界回退（只回退，不前进）
            window = text[start:end]
            for sep in ["。", "！", "？", ".", "!", "?"]:
                pos = window.rfind(sep)
                if pos != -1 and pos > len(window) * 0.6:
                    end = start + pos + 1
                    break

            # 3. 兜底：保证 end 一定前进
            if end <= start:
                end = min(start + self.chunk_size, text_len)

            chunk_text = text[start:end]

            chunks.append(
                TextChunk(
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_id": chunk_id,
                        "char_start": start,
                        "char_end": end,
                    },
                )
            )

            if len(chunks) >= 2:
                prev = chunks[-2]
                curr = chunks[-1]

                # 如果当前 chunk 没有引入任何新字符
                if curr.metadata["char_end"] <= prev.metadata["char_end"]:
                    chunks.pop()
                    break

            chunk_id += 1

            # 4. 计算下一轮 start，确保单调递增
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = end

            start = next_start

        return chunks
```



### 总结

```
TextSplitter 是“结构工程模块”，不是“智能模块”。
它的成功标准是：稳定、可追溯、可预测，而不是语义完美。
```



## 测试数据

```
这是一个用于测试 Text Processor 的中英混合文本示例。

本系统的目标是构建一个垂直领域的智能问答系统，支持文档加载、文本切分、向量检索以及基于大模型的回答生成。
在实际应用中，文档内容往往比较长，如果不进行合理切分，会严重影响检索效果。

The system supports multiple document formats, including PDF, Markdown, and TXT.
Each document will be split into smaller chunks for embedding and retrieval.
This process is commonly known as Retrieval-Augmented Generation (RAG).

在 RAG 系统中，chunk size 是一个非常关键的参数。
如果 chunk 太小，语义不完整；如果 chunk 太大，又会引入噪声。
Therefore, choosing an appropriate chunk size is essential for good performance.

本测试文本还包含一些较长的连续内容，用于触发基于字符长度的切分逻辑。
我们希望验证在超过设定的 chunk_size（例如 500 字符）时，系统是否能够正确地进行切分，
并且在切分时尽量保证语义完整性，而不是简单地在字符边界处强行截断。

Finally, this paragraph is written entirely in English to ensure that the sentence-based splitting
logic works correctly across different languages and punctuation styles.
```



## 模块 2 验收测试

```
src/test_processor.py
```



```
from pathlib import Path
from src.loaders.txt_loader import TxtLoader
from src.loaders.md_loader import MarkdownLoader
from src.loaders.pdf_loader import PdfLoader
from src.processors.cleaner import TextCleaner
from src.processors.splitter import TextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent

cleaner = TextCleaner()
splitter = TextSplitter()


def txt_test():
    loader = TxtLoader()
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.txt")

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
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.md")

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
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.pdf")

    for doc in docs:
        doc.content = cleaner.clean(doc.content)
        chunks = splitter.split(doc)

        print(f"Chunks: {len(chunks)}")
        for c in chunks:
            print("----")
            print(c.content)
            print(c.metadata)

if __name__ == "__main__":
    txt_test()
    md_test()
    pdf_test()
```



## 测试结果

### txt和md文件切割结果

![model_2_2](md_img/model_2_2.png)



### pdf文件切割结果

![model_2_3](md_img/model_2_3.png)



## 注意事项



### 问题一：中英混合文本导致语义断裂（在pdf出现）

#### 现象

```
英文单词之间没有空格，语义 embedding 质量明显下降
如：Thesystemsupportsmultipledocumentformats
```



#### 产生原因

```
PDF / OCR / 某些解析器会：删除英文词间空格或合并跨行英文
```



### 问题二：严格 chunk_size 导致语义被强行截断

#### 现象

```
句子在 500 字边界被硬切，chunk 尾部 / 头部语义不完整
如一个chunk切割内容如下，可以发现句子被截断
tion(RAG). 在 RAG 系统中，chunksize 是一个非常关键的参数。 如果 chunk 太小，语义不完整；如果 chunk 太大，又会引入噪声。
```



#### 产生原因

```
只按字符长度切，不考虑：句号，语义停顿
向量模型对半句话 embedding 很差
```



#### 解决办法

```
模块2暂时不解决这个问题
```



#### 结论

```
chunk_size 是“上限”，不是“目标”
真正的最小单位应是：完整句子
```



### 问题三：overlap 引入语义重复 & 噪声

#### 现象

```
chunk 之间内容重复
检索结果返回“几乎相同”的段落
```



#### 产生原因

```
overlap 是按字符回退
```



#### 经验结论

```
overlap 的目标是：防止信息丢失，不是防止重复
重复是可以容忍的，丢语义是不可接受的
```



### 问题四：尾 chunk 过短（纯 overlap chunk）

#### 经验结论

```
最后一个 chunk 允许“不优雅”，但必须“正确”
```



### 问题五：切割出现死循环

#### 产生原因

```
切分逻辑里存在「start 不前进」的路径。
```



#### 经验结论

```
1. 所有切分循环必须保证索引单调递增
2. 任意语义优化都必须有硬兜底路径
3. overlap 不得影响主游标
4. 永远允许“不优雅的最后一个 chunk”
```



# 模块 3：向量化器（Embedding Model）



## 模块定位

- 接收 模块 2 输出的 **TextChunk**
- 调用 **OpenAI Embedding API**
- 批量 / 异步生成向量
- 输出 **Embedding 结果 + 原 metadata（只追加字段）**



**不承担的职责**

- 不存储向量
- 不计算相似度
- 不做检索
- 不排序
- 不做 LangChain / LangSmith
- 不关心 Prompt / LLM



## 增加依赖（requirements.txt）：

```
# -----------------------------
# 模块 1：Document Loader 依赖
# -----------------------------
pdfplumber>=0.10.0

# -----------------------------
# 模块 2：TextProcessor / TextSplitter 依赖
# -----------------------------
# 纯 Python，无新增依赖（已包含在 Python 标准库）

# -----------------------------
# 模块 3：Embedding Model 依赖
# -----------------------------
numpy>=1.24
openai>=1.28.0

# -----------------------------
# 可选 / 工程化建议
# -----------------------------
# python-dotenv 可选，用于管理 OPENAI_API_KEY
python-dotenv>=1.0.0
# LangChain / LangSmith（实验 & 观测层）
langchain>=0.1.16
langchain-openai>=0.0.6
langsmith>=0.1.0
```



## **控制台运行指令**

```
pip install -r requirements.txt
```



## 输入

```
List[TextChunk]
```



## 输出

```
EmbeddedChunk {
    vector: np.ndarray
    metadata: dict
}

metadata：{
    "embedding_model": "text-embedding-3-small",
    "embedding_dim": 1536
}
```



## 项目结构

```
src/
├─ embeddings/
    ├─ __init__.py
    ├─ base.py                 # Embedding 抽象接口
    ├─ embedded_chunk.py       # Embedding 输出数据结构
    └─ openai_embedding.py     # OpenAI Embedding 实现
```



![model_3_1](md_img/model_3_1.png)



## 配置文件（需要自己去官网生成密钥）

```
LangSmith.env
```



```
LANGSMITH_TRACING=
LANGSMITH_ENDPOINT=""
LANGSMITH_API_KEY=""
LANGSMITH_PROJECT=""
OPENAI_API_KEY=""
OPENAI_BASE_URL=""
```



## 核心数据结构

```
src/embeddings/embedded_chunk.py # Embedding 输出数据结构
```



```
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class EmbeddedChunk:
    vector: np.ndarray # embedding 向量（维度由模型决定）
    metadata: Dict # TextChunk.metadata
```



## Embedding 抽象接口（可扩展核心）

```
src/embeddings/base.py # Embedding 抽象接口
```



```
from abc import ABC, abstractmethod
from typing import List
from ..processors.chunk import TextChunk
from .embedded_chunk import EmbeddedChunk


class BaseEmbeddingModel(ABC):

	# async 接口冻结，未来可无缝替换为本地模型 / 其他云模型
    @abstractmethod
    async def embed(
        self, chunks: List[TextChunk]
    ) -> List[EmbeddedChunk]:
        """
        将 TextChunk 批量转换为 EmbeddedChunk（异步）
        """
        pass
```



## OpenAI Embedding 实现（工程级）

```
src/embeddings/openai_embedding.py # OpenAI Embedding 实现
```



```
import os
import numpy as np
from typing import List
from openai import AsyncOpenAI

from ..processors.chunk import TextChunk
from .base import BaseEmbeddingModel
from .embedded_chunk import EmbeddedChunk


class OpenAIEmbedding(BaseEmbeddingModel):
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        client: AsyncOpenAI | None = None,
    ):
        self.model_name = model_name
        self.client = client or AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    async def embed(
        self, chunks: List[TextChunk]
    ) -> List[EmbeddedChunk]:

        if not chunks:
            return []

        texts = [c.content for c in chunks]

        # === 核心：一次性批量 embedding ===
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )

        vectors = [
            np.array(item.embedding, dtype=np.float32)
            for item in response.data
        ]

        results: List[EmbeddedChunk] = []

        for chunk, vector in zip(chunks, vectors):
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "embedding_model": self.model_name,
                    "embedding_dim": vector.shape[0],
                }
            )

            results.append(
                EmbeddedChunk(
                    vector=vector,
                    metadata=metadata,
                )
            )

        return results
```



## 模块 3 验收测试

```
src/test_embedding.py
```



```
import asyncio
from processors.chunk import TextChunk
from embeddings.openai_embedding import OpenAIEmbedding


async def test_embedding():
    chunks = [
        TextChunk(
            content="这是一个用于测试 embedding 的文本。",
            metadata={
                "source": "test.txt",
                "file_type": "txt",
                "chunk_id": 0,
                "char_start": 0,
                "char_end": 18,
            },
        ),
        TextChunk(
            content="This is another test chunk written in English.",
            metadata={
                "source": "test.txt",
                "file_type": "txt",
                "chunk_id": 1,
                "char_start": 19,
                "char_end": 60,
            },
        ),
    ]

    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    assert len(embedded_chunks) == 2
    assert embedded_chunks[0].vector.ndim == 1
    assert "embedding_model" in embedded_chunks[0].metadata

    print("Embedding dim:", embedded_chunks[0].vector.shape[0])
    print("Metadata:", embedded_chunks[0].metadata)


if __name__ == "__main__":
    asyncio.run(test_embedding())
```



## 测试结果

![model_3_2](md_img/model_3_2.png)



# 模块4：向量数据库 (Vector Database) 



## 模块定位

**存储**

- 存储 `EmbeddedChunk`
- 向量 + metadata 一一对应

**相似度检索**

- 输入：query vector
- 输出：Top-K `EmbeddedChunk`（带 score）

**CRUD（最小集）**

- add
- search
- clear（测试用）



**模块 4 明确不做**

-  不调用 OpenAI
-  不生成 embedding
-  不做 query → embedding
-  不排序以外的业务逻辑
-  不拼 Prompt

 

## 输入

```
EmbeddedChunk {
    vector: np.ndarray      # shape = (dim,)
    metadata: dict          # 含 source / chunk_id / embedding_model 等
}
```



## 输出

```
SearchResult {
    chunk: EmbeddedChunk
    score: float            # 相似度（cosine）
}
```



## 项目结构

```
src/
├─ vectorstores/
    ├─ __init__.py
    ├─ base.py              # VectorStore 抽象接口
    ├─ search_result.py     # 检索结果数据结构
    └─ memory_store.py     # In-Memory 实现
```



![model_4_1](md_img/model_4_1.png)



## 核心数据结构

```
src/vectorstores/search_result.py # 检索结果数据结构
```



```
from dataclasses import dataclass
from typing import Any

@dataclass
class SearchResult:
    chunk: Any      # EmbeddedChunk
    score: float
```



## vectorstore抽象接口

```
src/vectorstores/base.py # VectorStore 抽象接口
```



```
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from .search_result import SearchResult
from src.embeddings.embedded_chunk import EmbeddedChunk


class BaseVectorStore(ABC):

    @abstractmethod
    def add(self, chunks: List[EmbeddedChunk]) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List[SearchResult]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass
```



## In-Memory 向量库

```
src/vectorstores/memory_store.py # In-Memory 实现（第一版）
```



```
import numpy as np
from typing import List
from .base import BaseVectorStore
from .search_result import SearchResult
from src.embeddings.embedded_chunk import EmbeddedChunk


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class InMemoryVectorStore(BaseVectorStore):

    def __init__(self):
        self._chunks: List[EmbeddedChunk] = []

    def add(self, chunks: List[EmbeddedChunk]) -> None:
        self._chunks.extend(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List[SearchResult]:

        results: List[SearchResult] = []

        for chunk in self._chunks:
            score = cosine_similarity(query_vector, chunk.vector)
            results.append(SearchResult(chunk=chunk, score=score))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        self._chunks.clear()
```



## 模块 4 验收测试

```
src/test_vectorstore.py
```



```
import asyncio
from pathlib import Path

import os
from dotenv import load_dotenv

# 加载 .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "LangSmith.env")

from src.loaders.md_loader import MarkdownLoader
from src.loaders.pdf_loader import PdfLoader
from src.loaders.txt_loader import TxtLoader
from src.processors.cleaner import TextCleaner
from src.processors.splitter import TextSplitter
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.vectorstores.memory_store import InMemoryVectorStore


cleaner = TextCleaner()
splitter = TextSplitter()


def build_chunks_from_docs(docs):
    all_chunks = []
    for doc in docs:
        doc.content = cleaner.clean(doc.content)
        chunks = splitter.split(doc)
        all_chunks.extend(chunks)
    return all_chunks


def txt_test():
    loader = TxtLoader()
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.txt")
    return build_chunks_from_docs(docs)


def md_test():
    loader = MarkdownLoader()
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.md")
    return build_chunks_from_docs(docs)


def pdf_test():
    loader = PdfLoader()
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.pdf")
    return build_chunks_from_docs(docs)


async def test_vector_store(chunks):
    # 1. 文档 → chunks
    print(f"Chunks: {len(chunks)}")

    # 2. chunks → embeddings
    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    # 3. 向量库
    store = InMemoryVectorStore()
    store.add(embedded_chunks)

    # 4. 用第一个 chunk 向量当 query
    query_vector = embedded_chunks[0].vector
    results = store.search(query_vector, top_k=3)

    print(f"Search results: {len(results)}")
    for r in results:
        print("----")
        print(f"Score: {r.score:.4f}")
        print(r.chunk.metadata)


if __name__ == "__main__":
    chunks = md_test()
    # chunks = txt_test()
    # chunks = pdf_test()

    asyncio.run(test_vector_store(chunks))
```

 

## 测试结果

![model_4_2](md_img/model_4_2.png)

 

# 模块5：检索器 (Retriever) 



## 模块定位

```
把“用户问题（query text）” → 变成“最相关的 chunks”

query_text
 → embedding(query_text)
 → vectorstore.search
 → Top-K EmbeddedChunk
```



## 输入

```
query: str
```



## 输出

```
List[SearchResult]
```



## 项目结构

```
src/
├─ retrievers/
    ├─ __init__.py
    ├─ base.py               # Retriever 抽象接口
    └─ simple_retriever.py   # 最小可用实现
```



![model_5_1](md_img/model_5_1.png)



## retriever抽象接口

### 注意

Retriever **不关心 embedding 维度**

Retriever **不关心 chunk 内容**

Retriever **不关心 metadata 结构**



```
src/retrievers/base.py # Retriever 抽象接口
```



```
from abc import ABC, abstractmethod
from typing import List
from src.vectorstores.search_result import SearchResult


class BaseRetriever(ABC):

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        根据 query 返回 Top-K 相关 chunk
        """
        pass
```



## 模块3补充

### 原因

补充一个 **query embedding 接口**，否则 Retriever 会“硬塞 chunk”



### 接口定义

```
src/embeddings/base.py
```



```
# 新增接口
class BaseEmbedding(ABC):

    @abstractmethod
    async def embed(
            self,
            chunks
    ) -> List[EmbeddedChunk]:
        pass

    @abstractmethod
    async def embed_texts(
            self,
            texts: List[str]
    ) -> List[EmbeddedChunk]:
        """
        用于 query embedding
        """
        pass
```



### 接口实现

```
src/embeddings/openai_embedding.py
```



```
# 新增实现
    async def embed_texts(self, texts: List[str]):
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )

        results = []
        for text, item in zip(texts, response.data):
            results.append(
                EmbeddedChunk(
                    vector=np.array(item.embedding),
                    metadata={
                        "embedding_model": self.model_name,
                        "embedding_dim": len(item.embedding),
                    }
                )
            )
        return results
```



## 实现检索

```
src/retrievers/simple_retriever.py # 最小可用实现
```



```
from typing import List
from src.retrievers.base import BaseRetriever
from src.vectorstores.base import BaseVectorStore
from src.vectorstores.search_result import SearchResult
from src.embeddings.base import BaseEmbedding


class SimpleRetriever(BaseRetriever):

    def __init__(
        self,
        embedder: BaseEmbedding,
        vectorstore: BaseVectorStore,
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
```



## 模块5验收测试

```
src/test_retriever.py
```



```
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from src.loaders.pdf_loader import PdfLoader
from src.loaders.txt_loader import TxtLoader

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "LangSmith.env")

from src.loaders.md_loader import MarkdownLoader
from src.processors.cleaner import TextCleaner
from src.processors.splitter import TextSplitter
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.vectorstores.memory_store import InMemoryVectorStore
from src.retrievers.simple_retriever import SimpleRetriever


cleaner = TextCleaner()
splitter = TextSplitter()


def build_chunks_from_docs(docs):
    all_chunks = []
    for doc in docs:
        doc.content = cleaner.clean(doc.content)
        chunks = splitter.split(doc)
        all_chunks.extend(chunks)
    return all_chunks


def txt_test():
    loader = TxtLoader()
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.txt")
    return build_chunks_from_docs(docs)


def md_test():
    loader = MarkdownLoader()
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.md")
    return build_chunks_from_docs(docs)


def pdf_test():
    loader = PdfLoader()
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.pdf")
    return build_chunks_from_docs(docs)


async def test_retriever(chunks):
    # 1. 文档 → chunks
    print(f"Chunks: {len(chunks)}")

    # 2. chunks → embeddings
    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    # 3. 向量库
    store = InMemoryVectorStore()
    store.add(embedded_chunks)

    # 4. Retriever
    retriever = SimpleRetriever(
        embedder=embedder,
        vectorstore=store
    )

    query = "什么是 RAG 系统？"
    results = await retriever.retrieve(query, top_k=3)

    print(f"Query: {query}")
    for r in results:
        print("----")
        print(f"Score: {r.score:.4f}")
        print(r.chunk.metadata)


if __name__ == "__main__":
    chunks = md_test()
    # chunks = txt_test()
    # chunks = pdf_test()
    
    asyncio.run(test_retriever(chunks))
```



## 测试结果

![model_5_1](md_img/model_5_2.png)



# 模块6：提示词构造器（PromptBuilder）



## 模块定位

PromptBuilder 是 **RAG 链路中的“组装层”**，负责把：

- 用户 Query
- Retriever（/Reranker）返回的 Top-K Chunks

组织成 **可控、稳定、可解释** 的 Prompt，交给 LLM 使用。

**重要原则**：PromptBuilder **不做检索、不调用模型、不做推理**，只做结构化拼装。



## 输入

```
query: str
chunks: List[EmbeddedChunk | SearchResult]
```



## 输出

```
prompt: str
```



## 标准 RAG Prompt 结构

```
你是一个严谨的智能问答助手，请基于给定的上下文回答问题。


【上下文】
[1]
...


[2]
...


【问题】
...


【要求】
- 仅基于上下文回答
- 如果无法从上下文得到答案，请回答“未在文档中找到相关信息”
```



## 项目结构

```
src/
└─ prompts/
	├─ __init__.py
	├─ base.py # PromptBuilder 抽象接口
	├─ simple_prompt.py # 最小可用实现
```



![model_6_1](md_img/model_6_1.png)



## PromptsBuilder 抽象接口

```
src/prompts/base.py # PromptBuilder 抽象接口
```



```
from abc import ABC, abstractmethod
from typing import List


class BasePromptBuilder(ABC):

    @abstractmethod
    def build(
            self,
            query: str,
            contexts: List[str]
    ) -> str:
        """
        根据 query 和上下文内容构建 prompt
        """

    pass
```



## prompt实现

```
src/prompts/simple_prompt.py # 最小可用实现
```



```
from typing import List
from .base import BasePromptBuilder


class SimplePromptBuilder(BasePromptBuilder):
    def build(
        self, 
        query: str, 
        contexts: List[str] 
    ) -> str:
        context_text = "\n\n".join(
            f"[{i + 1}] {c}" for i, c in enumerate(contexts)
        )

        prompt = f"""
        你是一个严谨的智能问答助手，请基于给定的上下文回答问题。
    
        【上下文】
        {context_text}
    
        【问题】
        {query}
    
        【要求】
        - 仅基于上下文回答
        - 如果无法从上下文得到答案，请回答“未在文档中找到相关信息”
        """.strip()

        return prompt
```



## 模块6验收测试

```
src/test_prompt.py
```



```
from src.prompts.simple_prompt import SimplePromptBuilder


def test_prompt_builder():
    query = "什么是 RAG 系统？"
    contexts = [
        "RAG 是一种结合检索与生成的技术。",
        "RAG 系统通过向量检索增强大模型回答能力。"
    ]

    builder = SimplePromptBuilder()
    prompt = builder.build(query, contexts)

    print(prompt)


if __name__ == "__main__":
    test_prompt_builder()
```

 

## 测试结果

![model_6_2](md_img/model_6_2.png)



# 模块7 ：回答生成器（LLM Client）

## 模块定位

只做一件事：Prompt(str) → LLM → Answer(str + metadata)



**模块 7 不做**

- 不做检索
- 不做 prompt 拼接
- 不做 rerank
- 不解析 chunk
- 不缓存



## 输入

```
prompt: str # 来自模块 7（PromptBuilder）的完整 Prompt
```



## 输出

```
LLMAnswer {
    content: str        # 模型生成的回答
    metadata: dict      # 模型信息、token、耗时等
}
```



## 项目结构

```
src/
├─ llms/
    ├─ __init__.py
    ├─ base.py              # LLM 抽象接口
    ├─ answer.py            # LLM 输出数据结构
    └─ openai_chat.py       # OpenAI Chat 实现
```



![model_7_1](md_img/model_7_1.png)



## 核心数据结构

```
src/llms/answer.py # LLM 输出数据结构
```



```
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class LLMAnswer:
    content: str
    metadata: Dict[str, Any]
```



## LLM 抽象接口

```
src/llms/base.py
```



```
from abc import ABC, abstractmethod
from .answer import LLMAnswer


class BaseLLM(ABC):

    @abstractmethod
    async def generate(self, prompt: str) -> LLMAnswer:
        """
        输入 Prompt，返回模型生成的回答
        """
        pass
```



## OpenAI Chat 实现

```
src/llms/openai_chat.py # OpenAI Chat 实现
```



```
import time
from openai import AsyncOpenAI
from .base import BaseLLM
from .answer import LLMAnswer


class OpenAIChatLLM(BaseLLM):

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ):
        self.client = AsyncOpenAI()
        self.model = model
        self.temperature = temperature

    async def generate(self, prompt: str) -> LLMAnswer:
        start_time = time.time()

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )

        elapsed = time.time() - start_time

        message = response.choices[0].message.content

        return LLMAnswer(
            content=message,
            metadata={
                "model": self.model,
                "temperature": self.temperature,
                "elapsed_time": round(elapsed, 3),
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        )
```



## 验收测试

```
src/test_llm.py
```



```
import asyncio
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "LangSmith.env")

from src.llms.openai_chat import OpenAIChatLLM


PROMPT = """
你是一个严谨的智能问答助手，请基于给定的上下文回答问题。

【上下文】
[1] RAG 是一种结合检索与生成的技术。
[2] RAG 系统通过向量检索增强大模型回答能力。

【问题】
什么是 RAG 系统？

【要求】
- 仅基于上下文回答
- 如果无法从上下文得到答案，请回答“未在文档中找到相关信息”
"""


async def test_llm():
    llm = OpenAIChatLLM()
    answer = await llm.generate(PROMPT)

    print("Answer:")
    print(answer.content)
    print("----")
    print("Metadata:")
    for k, v in answer.metadata.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    asyncio.run(test_llm())
```



## 测试结果

![model_7_2](md_img/model_7_2.png)



# 开发模块8前重要修改！！！

## 说明

因为RAG core需要结合前面所有模块，有些模块原来没有考虑模块8的应用，故在开发模块8之前需要修改。



## 模块3修改

### 核心数据结构

```
src/embeddings/embedded_chunk.py
```



```
@dataclass
class EmbeddedChunk:
    content: str  # 新增，chunk 原文
    vector: np.ndarray # embedding 向量（维度由模型决定）
    metadata: Dict # TextChunk.metadata
```



### OpenAI Embedding 实现

```
src/embeddings/openai_embedding.py
```



```
# embed方法中修改
results.append(
    EmbeddedChunk(
        content=chunk.content, # 修改这里
        vector=vector,
        metadata=metadata,
    )
)

# embed_texts方法中修改
for text, item in zip(texts, response.data):
    results.append(
        EmbeddedChunk(
            content=text, # 添加内容
            vector=np.array(item.embedding),
            metadata={
            "embedding_model": self.model_name,
            "embedding_dim": len(item.embedding),
        }
    )
)

```



### 验收测试代码与测试结果与原来一样



## 模块6修改

### PromptsBuilder 抽象接口

```
src/prompts/base.py
```



```
class BasePromptBuilder(ABC):

    @abstractmethod
    def build(
            self,
            query: str,
            chunks: List[SearchResult] # 修改这里
    ) -> str:
        """
        根据 query 和上下文内容构建 prompt
        """

    pass
```



### prompt实现

```
src/prompts/simple_prompt.py
```



```
from typing import List
from .base import BasePromptBuilder
from src.vectorstores.search_result import SearchResult


class SimplePromptBuilder(BasePromptBuilder):

    def build(
        self,
        query: str,
        chunks: List[SearchResult] # 修改这里
    ) -> str:

        context_text = "\n\n".join(
            f"[{i + 1}] {r.chunk.content}"
            for i, r in enumerate(chunks)
        )

        prompt = f"""
你是一个严谨的智能问答助手，请基于给定的上下文回答问题。

【上下文】
{context_text}

【问题】
{query}

【要求】
- 仅基于上下文回答
- 如果无法从上下文得到答案，请回答“未在文档中找到相关信息”
""".strip()

        return prompt
```



### 验收测试

```
src/test_prompt.py
```



```
from src.prompts.simple_prompt import SimplePromptBuilder
from src.vectorstores.search_result import SearchResult
from src.embeddings.embedded_chunk import EmbeddedChunk
import numpy as np


def test_prompt_builder():
    query = "什么是 RAG 系统？"

    # 构造假的 EmbeddedChunk（向量在这里不重要）
    chunk1 = EmbeddedChunk(
        content="RAG 是一种结合检索与生成的技术。",
        vector=np.zeros(3),
        metadata={}
    )

    chunk2 = EmbeddedChunk(
        content="RAG 系统通过向量检索增强大模型回答能力。",
        vector=np.zeros(3),
        metadata={}
    )

    # 构造 SearchResult
    results = [
        SearchResult(chunk=chunk1, score=0.9),
        SearchResult(chunk=chunk2, score=0.8),
    ]

    builder = SimplePromptBuilder()
    prompt = builder.build(query=query, chunks=results)

    print(prompt)


if __name__ == "__main__":
    test_prompt_builder()
```



### 测试结果

![model_6_3](md_img/model_6_3.png)



# 模块8 ：系统编排层（RAG Core）

## 模块定位

**把已有模块按固定顺序串起来**

- 组合调用模块
- 控制数据流顺序
- 汇总最终输出
- 插入观测（LangSmith）



**模块 9 明确禁止**

- 不写 embedding 逻辑
- 不写检索算法
- 不写 prompt 规则
- 不直接调用 OpenAI SDK
- 不修改 chunk / metadata 结构



## 输入

```
query: str
```



## 输出

```
RAGResult {
    answer: str
    sources: List[TextChunk]
    metadata: dict
}
```



## 项目结构

```
src/
├─ rag/
    ├─ __init__.py
    ├─ result.py           # RAG 输出结构
    └─ rag_core.py         # RAG 主流程
```



![model_8_1](md_img/model_8_1.png)



## 核心数据结构

```
src/rag/result.py # RAG 输出结构
```



```
from dataclasses import dataclass
from typing import List, Dict, Any
from src.processors.chunk import TextChunk


@dataclass
class RAGResult:
    answer: str
    sources: List[TextChunk]
    metadata: Dict[str, Any]
```



## RAG Core 实现（核心）

```
src/rag/rag_core.py # RAG 主流程
```



```
import time
from typing import List

# 添加条件导入以提高健壮性
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    # 创建一个空的装饰器作为替代
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from src.embeddings.base import BaseEmbedding
from src.vectorstores.base import BaseVectorStore
from src.retrievers.base import BaseRetriever
from src.prompts.base import BasePromptBuilder
from src.llms.base import BaseLLM
from .result import RAGResult


class RAGCore:

    def __init__(
        self,
        embedder: BaseEmbedding,
        vectorstore: BaseVectorStore,
        retriever: BaseRetriever,
        prompt_builder: BasePromptBuilder,
        llm: BaseLLM,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.top_k = top_k

    @traceable(name="rag_answer")
    async def answer(self, query: str) -> RAGResult:

        start_time = time.time()

        # 1. 检索相关 chunks
        retrieved_chunks = await self.retriever.retrieve(
            query=query,
            top_k=self.top_k
        )

        # 2. 构建 Prompt
        prompt = self.prompt_builder.build(
            query=query,
            chunks=retrieved_chunks
        )

        # 3. LLM 生成回答
        llm_answer = await self.llm.generate(prompt)

        elapsed = round(time.time() - start_time, 3)

        return RAGResult(
            answer=llm_answer.content,
            sources=retrieved_chunks,
            metadata={
                "latency": elapsed,
                "top_k": self.top_k,
                "llm": llm_answer.metadata,
            }
        )
```



**重要设计**

- RAGCore **只依赖抽象**
- 所有实现可替换



## 验收测试

```
src/test_rag.py
```



```
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from src.loaders.pdf_loader import PdfLoader
from src.loaders.txt_loader import TxtLoader

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "LangSmith.env")

from src.loaders.md_loader import MarkdownLoader
from src.processors.cleaner import TextCleaner
from src.processors.splitter import TextSplitter

from src.embeddings.openai_embedding import OpenAIEmbedding
from src.vectorstores.memory_store import InMemoryVectorStore
from src.retrievers.simple_retriever import SimpleRetriever
from src.prompts.simple_prompt import SimplePromptBuilder
from src.llms.openai_chat import OpenAIChatLLM
from src.rag.rag_core import RAGCore

cleaner = TextCleaner()
splitter = TextSplitter()


def build_chunks_from_docs(docs):
    """模块 1 + 模块 2 组合，仅用于测试"""
    all_chunks = []
    for doc in docs:
        doc.content = cleaner.clean(doc.content)
        chunks = splitter.split(doc)
        all_chunks.extend(chunks)
    return all_chunks


def txt_test():
    loader = TxtLoader()
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.txt")
    return build_chunks_from_docs(docs)


def md_test():
    loader = MarkdownLoader()
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.md")
    return build_chunks_from_docs(docs)


def pdf_test():
    loader = PdfLoader()
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.pdf")
    return build_chunks_from_docs(docs)


async def test_rag(chunks):
    # 1. 构建文档 chunks
    print(f"Total chunks: {len(chunks)}")

    # 2. Embedding + VectorStore
    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    store = InMemoryVectorStore()
    store.add(embedded_chunks)

    retriever = SimpleRetriever(vectorstore=store, embedder=embedder)

    # 3. Prompt + LLM
    prompt_builder = SimplePromptBuilder()
    llm = OpenAIChatLLM()

    rag = RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        top_k=3,
    )

    # 4. 问答
    result = await rag.answer("什么是 RAG 系统？")

    print("Answer:")
    print(result.answer)
    print("----")
    print("Sources:")
    for s in result.sources:
        print("----")
        print(f"Score: {s.score:.4f}")
        print("Content:")
        print(s.chunk.content)
        print("Metadata:")
        print(s.chunk.metadata)


if __name__ == "__main__":
    chunks = md_test()
    # chunks = txt_test()
    # chunks = pdf_test()

    asyncio.run(test_rag(chunks))
```



## 测试结果

### 控制台结果

![model_8_2](md_img/model_8_2.png)



### LangSmith 结果

![model_8_3](md_img/model_8_3.png)



-----------------------------------------------------------------------------------**图片分割线**----------------------------------------------------------------------------------------



![model_8_4](md_img/model_8_4.png)



# 模块9：重排序器（Reranker）

## 模块定位

模块 9 的目标是在已有 **向量召回（Retriever）** 的基础上，引入 **Reranker（二次排序）**，验证其在 RAG 系统中的实际价值，并分析不同 reranking 方法的能力边界。



**RAG 流程位置**

```
# Reranker 位于 初始向量召回之后、Prompt 构造之前。
Query
↓
Embedding
↓
VectorStore Top-K Recall ←（模块 5）
↓
Reranker（二次排序） ←（模块 9）
↓
Prompt 构造
↓
LLM 生成答案
```



## 输入

```
query_vector: np.ndarray
results: List[SearchResult]
top_k: int
```



## 输出

```
List[SearchResult]   # 按 rerank score 降序
```



## 项目结构

```
src/
├── rerankers/
    ├── __init__.py
    ├── base.py
    └── embedding_reranker.py   
```



![model_9_1](md_img/model_9_1.png)



## rerankers抽象接口

- 不关心 embedding
- 不关心 vectorstore
- 只关心 **query + chunk 内容**

```
src/rerankers/base.py
```



```
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from src.vectorstores.search_result import SearchResult


class BaseReranker(ABC):

    @abstractmethod
    def rerank(
        self,
        query_vector: np.ndarray,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        对初始检索结果进行二次排序
        """
        pass
```



## 实现重排序

由于 OpenAI 官方 Rerank API 在当前环境不可用，本模块采用：

**EmbeddingReranker（基于 embedding 的 rerank）**



**其核心思想是：**

- 使用与 Query 相同的 embedding 模型
- 重新计算 `query embedding` 与 `chunk embedding` 的相似度
- 对 Top-K 候选 chunk 进行重新排序



**设计定位：**

- 轻量级（Lightweight） reranker
- 不引入额外 LLM 调用
- 用于验证 reranking 思想，而非追求最优效果



```
src/rerankers/embedding_reranker.py
```



```
import numpy as np
from typing import List
from src.rerankers.base import BaseReranker
from src.vectorstores.search_result import SearchResult


class EmbeddingReranker(BaseReranker):
    """
    使用 query embedding 与 chunk embedding 的余弦相似度进行 rerank
    """

    def rerank(
        self,
        query_vector: np.ndarray,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:

        if not results:
            return []

        reranked = []

        for r in results:
            chunk_vector = r.chunk.vector
            score = self._cosine_similarity(query_vector, chunk_vector)

            reranked.append(
                SearchResult(
                    chunk=r.chunk,
                    score=score
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        )
```



## 在RAG Core 中实现“无侵入插入”和LangSmith

```
src/rag/rag_core.py
```



```
# 修改 RAGCore.__init__

def __init__(
        self,
        embedder: BaseEmbedding,
        vectorstore: BaseVectorStore,
        retriever: BaseRetriever,
        prompt_builder: BasePromptBuilder,
        llm: BaseLLM,
        reranker=None, # 新增（可选）
        top_k: int = 5,
    ):
    self.embedder = embedder
    self.vectorstore = vectorstore
    self.retriever = retriever
    self.prompt_builder = prompt_builder
    self.llm = llm
    self.reranker = reranker # 新增（可选）
    self.top_k = top_k
  
  
  
# 修改 RAGCore.answer

@traceable(name="rag_answer")
async def answer(self, query: str) -> RAGResult:

    start_time = time.time()

    # 1. 检索相关 chunks
    retrieved_chunks = await self.retriever.retrieve(
        query=query,
        top_k=self.top_k * 2   # rerank 前多取
    )

    # 2. Rerank（新增）
    if self.reranker:
        query_embedding = await self.retriever.embedder.embed_texts([query])
        query_vector = query_embedding[0].vector

        retrieved_chunks = self.reranker.rerank(
            query_vector=query_vector,
            results=retrieved_chunks,
            top_k=self.top_k
        )

    # 3. 构建 Prompt
    prompt = self.prompt_builder.build(
        query=query,
        chunks=retrieved_chunks
    )
```



## 验收测试

```
src/test_reranker.py
```



```
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from src.loaders.pdf_loader import PdfLoader
from src.loaders.txt_loader import TxtLoader
from src.rerankers.embedding_reranker import EmbeddingReranker

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "LangSmith.env")

from src.loaders.md_loader import MarkdownLoader
from src.processors.cleaner import TextCleaner
from src.processors.splitter import TextSplitter

from src.embeddings.openai_embedding import OpenAIEmbedding
from src.vectorstores.memory_store import InMemoryVectorStore
from src.retrievers.simple_retriever import SimpleRetriever
from src.prompts.simple_prompt import SimplePromptBuilder
from src.llms.openai_chat import OpenAIChatLLM
from src.rag.rag_core import RAGCore

cleaner = TextCleaner()
splitter = TextSplitter()


def build_chunks_from_docs(docs):
    """模块 1 + 模块 2 组合，仅用于测试"""
    all_chunks = []
    for doc in docs:
        doc.content = cleaner.clean(doc.content)
        chunks = splitter.split(doc)
        all_chunks.extend(chunks)
    return all_chunks


def txt_test():
    loader = TxtLoader()
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.txt")
    return build_chunks_from_docs(docs)


def md_test():
    loader = MarkdownLoader()
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.md")
    return build_chunks_from_docs(docs)


def pdf_test():
    loader = PdfLoader()
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.pdf")
    return build_chunks_from_docs(docs)


async def test_rag_compare(chunks,use_rerank: bool):
    # 1. 构建文档 chunks
    print(f"Total chunks: {len(chunks)}")

    # 2. Embedding + VectorStore
    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    store = InMemoryVectorStore()
    store.add(embedded_chunks)

    retriever = SimpleRetriever(vectorstore=store, embedder=embedder)

    # 3. Prompt + LLM + Reranker
    prompt_builder = SimplePromptBuilder()
    llm = OpenAIChatLLM()
    reranker = EmbeddingReranker() if use_rerank else None # 添加 Reranker

    rag = RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        reranker=reranker, # 添加 Reranker
        top_k=3,
    )

    # 4. 问答
    result = await rag.answer("什么是 RAG 系统？")

    print("Answer:")
    print(result.answer)
    print("----")
    print("Sources:")
    for s in result.sources:
        print("----")
        print(f"Score: {s.score:.4f}")
        print("Content:")
        print(s.chunk.content)
        print("Metadata:")
        print(s.chunk.metadata)

async def main():
    chunks = md_test()
    # chunks = txt_test()
    # chunks = pdf_test()

    print("不使用 Reranker")
    await test_rag_compare(chunks,use_rerank=False)
    print()
    print("---------------------------------------------------------------------------------------------------------")
    print("使用 Reranker")
    await test_rag_compare(chunks,use_rerank=True)

if __name__ == "__main__":

    asyncio.run(main())
```



## 测试结果

### 不使用 Reranker

![model_9_2](md_img/model_9_2.png)

### 使用 Reranker

![model_9_3](md_img/model_9_3.png)

### LangSmith 结果（同时生成2个）

![model_9_4](md_img/model_9_4.png)



## 结果说明

**当前这个测试用例里：使用 / 不使用 Reranker，结果几乎一致是“正常现象”，不是问题。**



### 结果一样的原因有三点

1. **候选 chunks 本来就只有 3 个**
2. **Retriever 使用的也是 embedding cosine**
3. **Reranker 用的还是同一套 embedding + cosine**



### 本质上

**现在是在用“同一个排序逻辑，对同一批数据，排序两次”**



### rerank 的价值前提

Rerank **不是用来优化 Top-3 的**，而是用来解决：**Top-50 / Top-100 中，前 5 名顺序不准的问题**



### EmbeddingReranker 适用场景

- chunk 语义单一
- 只有少数 chunk 包含真正答案
- 召回阶段噪声不严重



### EmbeddingReranker 无效场景

- chunk 内包含多个语义段
- 多个 chunk 同时包含答案句
- 需要判断“是否真正回答问题”而非“是否相关”



### 提高 rerank 价值的 2 个方法

**扩大初始召回数量**

**人为制造干扰 chunk**：在测试文档里加入与 RAG 无关但 embedding 相似的段落或关键词相同但语义不同的段落，rerank 会把语义真正回答问题的 chunk”拉到前面。



### 使用新测试例子的结果

**排序结果大多数情况下保持不变，score有所变化**

![model_9_5](md_img/model_9_5.png)

![model_9_6](md_img/model_9_6.png)



### 为什么排序没有明显变化？

原因并非 reranker 未生效，而是：**EmbeddingReranker 的能力上限被实验完整暴露出来**



核心原因：

1. 多个 chunk **同时包含答案句**
2. 每个 chunk 都在“讨论同一主题”
3. chunk 内部是「噪声 + 答案 + 无关内容」的混合体



在这种情况下：向量模型无法判断“哪个 chunk 更回答问题”

它只能判断：“这些 chunk 是否在谈论同一个语义空间”



### 与 LLM Reranker 的对比（方法论层面）

EmbeddingReranker 无法替代 LLM Reranker，只能作为补充方案



## 本模块最终结论

1. Reranking 是 RAG 系统中的关键第二阶段
2. EmbeddingReranker 在结构正确的 chunk 上可发挥一定作用
3. 当 chunk 质量不足时，reranking 无法挽救排序
4. 真正可靠的 reranking 需要 LLM 或 Cross-Encoder 模型



# 模块10：语义切分（Semantic Chunking）

## 模块定位

在前序模块中，文本切分主要基于 **字符长度 + 标点回退**，这在工程上是稳定的，但在语义层面存在天然上限：

- 一个 chunk 内可能包含 **多个主题**
- embedding 会被“平均化”，导致检索与 rerank 效果受限
- reranker 很难在“语义已混杂”的 chunk 上发挥作用



**模块 10 的目标**

在进入 embedding 之前，尽可能保证 **一个 chunk ≈ 一个语义单元**。



**流程位置**

```
[ Module 1 ] Document
↓
[ Module 2 ] Cleaner
↓
[ Module 2 ] Character / Sentence Splitter
↓
[ Module 10 ] Semantic Splitter ⭐
↓
[ Module 3 ] Embedding
↓
[ Module 4 ] VectorStore / [ Module 5 ] Retriever 及后续
```



## 输入

```
SemanticChunkerInput = {
    "text": str, # 原始文档全文（已读取）
    "metadata": dict, # 文档级元信息
}
```



## 输出

```
输出必须满足的三个不变量
	Chunk 内语义自洽
	Chunk 间语义边界清晰
	Chunk 长度稳定可控

SemanticChunk = {
    "content": str, # 语义自洽文本
    "chunk_id": int,
    "char_start": int,
    "char_end": int,
    "metadata": {
        "source": str,
        "file_type": str,
        "chunk_strategy": str, # e.g. semantic / char / sentence
	}
}

SemanticChunkerOutput = List[SemanticChunk]
```



## 项目结构

```
src/processors/semantic/
	├── _init_.py            
	├── base.py              # SemanticSplitter 接口
	├── heading_splitter.py  # 按 Markdown / 标题
	└── semantic_splitter.py # 组合调度器
```



![model_10_1](md_img/model_10_1.png)



## 抽象接口

```
src/processors/semantic/base.py
```



```
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
```



## 基于标题的语义切分

**核心思想**

- 以 `#`, `##`, `###` 作为**语义边界**

- 每个 section 作为一个语义 chunk

- section 内部再做轻量长度控制

  

```
src/processors/semantic/heading_splitter.py
```



```
import re
from typing import List
from src.processors.semantic.base import BaseSemanticSplitter
from src.processors.chunk import TextChunk
from src.loaders.document import Document


class HeadingSemanticSplitter(BaseSemanticSplitter):

    HEADING_PATTERN = re.compile(r"^(#{1,6})\\s+(.+)$", re.MULTILINE)

    def split(self, document: Document) -> List[TextChunk]:
        text = document.content
        matches = list(self.HEADING_PATTERN.finditer(text))

        if not matches:
            # 无标题，fallback
            return []

        chunks: List[TextChunk] = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue

            chunks.append(
                TextChunk(
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "semantic": "heading",
                        "heading": m.group(2),
                    }
                )
            )

        return  chunks
```



## 语义调度器（SemanticSplitter）

**职责**

- 判断文档类型
- 选择合适的 semantic splitter
- fallback 到模块 2 的 TextSplitter



```
src/processors/semantic/semantic_splitter.py
```



```
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
```



## 验收测试（对比4种情况）

```
src/test_semantic.py
```



```
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# ================== Path & Env ==================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "LangSmith.env")

# ================== Loaders ==================

from src.loaders.md_loader import MarkdownLoader
from src.loaders.txt_loader import TxtLoader
from src.loaders.pdf_loader import PdfLoader

# ================== Processors ==================

from src.processors.cleaner import TextCleaner

# 模块 2（baseline）
from src.processors.splitter import TextSplitter

# 模块 10（semantic）
from src.processors.semantic.semantic_splitter import SemanticSplitter

# ================== Embedding / Store ==================

from src.embeddings.openai_embedding import OpenAIEmbedding
from src.vectorstores.memory_store import InMemoryVectorStore

# ================== Retriever / Reranker ==================

from src.retrievers.simple_retriever import SimpleRetriever
from src.rerankers.embedding_reranker import EmbeddingReranker

# ================== Prompt / LLM / RAG ==================

from src.prompts.simple_prompt import SimplePromptBuilder
from src.llms.openai_chat import OpenAIChatLLM
from src.rag.rag_core import RAGCore


# ================== Split Builders ==================

cleaner = TextCleaner()


def build_chunks_text_splitter(docs):
    """模块 2：字符 / 规则切分（baseline）"""
    splitter = TextSplitter()
    all_chunks = []

    for doc in docs:
        doc.content = cleaner.clean(doc.content)
        all_chunks.extend(splitter.split(doc))

    return all_chunks


def build_chunks_semantic_splitter(docs):
    """模块 10：语义切分"""
    splitter = SemanticSplitter()
    all_chunks = []

    for doc in docs:
        doc.content = cleaner.clean(doc.content)
        all_chunks.extend(splitter.split(doc))

    return all_chunks


# ================== Data Loaders ==================

def load_md_chunks(use_semantic: bool):
    loader = MarkdownLoader()
    docs = loader.load(BASE_DIR / "data" / "model10" / "test.md")

    if use_semantic:
        print("[Semantic Splitter]")
        return build_chunks_semantic_splitter(docs)
    else:
        print("[Text Splitter]")
        return build_chunks_text_splitter(docs)


# ================== RAG Test ==================

async def test_rag(chunks, use_rerank: bool):
    print(f"Total chunks: {len(chunks)}")

    # 1. Embedding
    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    # 2. VectorStore
    store = InMemoryVectorStore()
    store.add(embedded_chunks)

    # 3. Retriever
    retriever = SimpleRetriever(
        embedder=embedder,
        vectorstore=store
    )

    # 4. RAG
    rag = RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=retriever,
        prompt_builder=SimplePromptBuilder(),
        llm=OpenAIChatLLM(),
        reranker=EmbeddingReranker() if use_rerank else None,
        top_k=5,
    )

    # 5. Ask
    result = await rag.answer(
        "在 RAG 系统中，Reranking 的作用是什么？"
    )

    print("Answer:")
    print(result.answer)
    print("----")
    print("Sources:")

    for s in result.sources:
        print("----")
        print(f"Score: {s.score:.4f}")
        print("Content:")
        print(s.chunk.content)
        print("Metadata:")
        print(s.chunk.metadata)


# ================== Main ==================

async def main():

    # 模块 2 + 无 Rerank
    print("\n==============================")
    print("TextSplitter + No Reranker")
    chunks = load_md_chunks(use_semantic=False)
    await test_rag(chunks, use_rerank=False)

    # 模块 2 + Rerank
    print("\n==============================")
    print("TextSplitter + Reranker")
    await test_rag(chunks, use_rerank=True)

    # 模块 10 + 无 Rerank
    print("\n==============================")
    print("SemanticSplitter + No Reranker")
    chunks = load_md_chunks(use_semantic=True)
    await test_rag(chunks, use_rerank=False)

    # 模块 10 + Rerank
    print("\n==============================")
    print("SemanticSplitter + Reranker")
    await test_rag(chunks, use_rerank=True)


if __name__ == "__main__":
    asyncio.run(main())
```



## 测试结果

### TextSplitter + No Reranker

![model_10_2](md_img/model_10_2.png)



### TextSplitter + Reranker

![model_10_3](md_img/model_10_3.png)



### SemanticSplitter + No Reranker

![model_10_4](md_img/model_10_4.png)



### SemanticSplitter + Reranker

![model_10_5](md_img/model_10_5.png)



### 结果对比

| Splitter         | Reranker | Top chunk 内容           | 分数差距                             |
| ---------------- | -------- | ------------------------ | ------------------------------------ |
| TextSplitter     | No       | Chunk 2/3 混合噪声与答案 | 分数接近，噪声占比高                 |
| TextSplitter     | Yes      | Chunk 3/6 真正答案       | Reranker 提升排序精度，答案排首位    |
| SemanticSplitter | No       | Chunk 2/3 混合噪声与答案 | 与 TextSplitter 差不多，分数略微不同 |
| SemanticSplitter | Yes      | Chunk 3/6 真正答案       | 明显提升，答案优先                   |



### 结果启示

**Reranker 必须在初始召回后使用**，尤其面对噪声多、关键词重复的文档。

**SemanticSplitter 更适合长文档或中英文混杂文档**，它保持 chunk 语义完整，避免噪声 chunk 过长或过短。

**对抗测试文档设计**：

- 多个噪声 chunk
- 关键词重复但语义无关
- 真正答案分散在中间
- 中英文混杂
   → 能够直观展示 Reranker 效果。



# 模块 11：答案聚合（Answer Aggregation）

## 模块定位

模块 11 的目标不是“再生成一次答案”，而是解决当前 RAG 系统的**三个结构性问题**：

1. **LLM 没有被强制消费检索 / 重排序结果**
2. **答案缺乏可解释性（不知道来自哪个 chunk）**
3. **无法为 UI / LangSmith / 评测提供标准结构**

**让 RAG 从「能答」升级为「可信 + 可解释 + 可评测」**



## 输入

```
query: str
results: List[SearchResult]

SearchResult 已包含：
chunk.content
chunk.metadata
score
（rerank 后的顺序）
```



## 输出

```
class AggregatedAnswer:
    answer: str
    evidence_chunks: List[EvidenceChunk]

class EvidenceChunk:
    rank: int
    score: float
    content: str
    source: Any
    chunk_id: int
```



## 项目结构

### 重要说明

从这里开始，测试文件全部移入test文件夹下，减少目录过于冗杂

```
测试文件中包含
BASE_DIR = Path(__file__).resolve().parent.parent

全部改为
BASE_DIR = Path(__file__).resolve().parent.parent.parent

模块1测试文件中
from loaders.txt_loader import TxtLoader
from loaders.md_loader import MarkdownLoader
from loaders.pdf_loader import PdfLoader

全部改为
from src.loaders.md_loader import MarkdownLoader
from src.loaders.pdf_loader import PdfLoader
from src.loaders.txt_loader import TxtLoader

即可正常运行各模块的测试代码
```



### 新结构如下

```
src/
└── aggregators/
    ├── __init__.py
    ├── base.py
    ├── schemas.py
    ├── simple_aggregator.py
    ├── answer_aware_aggregator.py

└── utils/
    ├── similarity.py.py
```



![model_11_1](md_img/model_11_1.png)



## 抽象接口

```
src/aggregators/base.py
```



```
from abc import ABC, abstractmethod
from typing import List
from src.vectorstores.search_result import SearchResult
from .schemas import AggregatedAnswer


class BaseAnswerAggregator(ABC):

    @abstractmethod
    async def aggregate(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        对检索到的 chunks 进行聚合 / 高亮 / 合并
        返回仍然是 SearchResult 列表
        """
        pass
```



## 数据结构定义

```
src/aggregators/schemas.py
```



```
from dataclasses import dataclass
from typing import List, Any


@dataclass
class EvidenceChunk:
    rank: int
    score: float
    content: str
    source: Any
    chunk_id: int


@dataclass
class AggregatedAnswer:
    answer: str
    evidence_chunks: List[EvidenceChunk]
```



## 核心实现

### 内部子结构

```
AnswerAwareAggregator
├── Input:
│   ├── query
│   ├── reranked SearchResult[]
│
├── Step 1: 生成“参考答案”（Answer Draft）
│   └── LLM(query + top chunks)
│
├── Step 2: Evidence Scoring
│   ├── 对每个 chunk 计算：
│   │   ├── 与 query 的相关度
│   │   ├── 与 answer draft 的一致度（关键）
│
├── Step 3: Evidence Selection
│   ├── 过滤噪声
│   ├── 重排
│   ├── 选 Top-N
│
├── Output:
│   └── AggregatedAnswer
│       ├── answer
│       └── evidence_chunks[]
```



### 代码

```
src/aggregators/simple_aggregator.py
```



```
from typing import List
from src.aggregators.base import BaseAnswerAggregator
from src.aggregators.schemas import EvidenceChunk
from src.vectorstores.search_result import SearchResult


class AnswerAggregator(BaseAnswerAggregator):

    def __init__(self, max_chunks: int = 3):
        self.max_chunks = max_chunks

    async def aggregate(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[EvidenceChunk]:
        """
        将 rerank 后的 SearchResult
        转换为最终对外展示的 EvidenceChunk
        """

        evidence_chunks: List[EvidenceChunk] = []

        for idx, r in enumerate(results[: self.max_chunks]):
            chunk = r.chunk

            evidence_chunks.append(
                EvidenceChunk(
                    rank=idx + 1,
                    score=r.score,
                    content=chunk.content,
                    source=chunk.metadata.get("source"),
                    chunk_id=chunk.metadata.get("chunk_id"),
                )
            )

        return evidence_chunks
```



## 修改模块8的RAGResult

```
src/rag/result.py
```



```
@dataclass
class RAGResult:
    answer: str
    sources: List[TextChunk]
    metadata: Dict[str, Any]
    evidence_chunks: Dict[str, Any] = None # 新增
```



## 将模块 11 接入现有 RAGCore

```
src/rag/rag_core.py
```



```
# 修改 RAGCore.__init__
def __init__(
    self,
    embedder: BaseEmbedding,
    vectorstore: BaseVectorStore,
    retriever: BaseRetriever,
    prompt_builder: BasePromptBuilder,
    llm: BaseLLM,
    reranker=None, 
    aggregator=None, # 新增（可选）
    top_k: int = 5,
):
    self.embedder = embedder
    self.vectorstore = vectorstore
    self.retriever = retriever
    self.prompt_builder = prompt_builder
    self.llm = llm
    self.reranker = reranker 
    self.aggregator = aggregator # 新增（可选）
    self.top_k = top_k

# 修改 RAGCore.answer
	@traceable(name="rag_answer")
    async def answer(self, query: str) -> RAGResult:

        start_time = time.time()

        # 1. 检索相关 chunks
        retrieved_chunks = await self.retriever.retrieve(
            query=query,
            top_k=self.top_k * 2   # rerank 前多取
        )

        # 2. Rerank（新增）
        if self.reranker:
            query_embedding = await self.retriever.embedder.embed_texts([query])
            query_vector = query_embedding[0].vector

            retrieved_chunks = self.reranker.rerank(
                query_vector=query_vector,
                results=retrieved_chunks,
                top_k=self.top_k
            )

        # 3. Aggregator（新增）
        evidence_chunks = []
        if self.aggregator:
            evidence_chunks = await self.aggregator.aggregate(
                query=query,
                results=retrieved_chunks
            )

        # 4. 构建 Prompt
        prompt = self.prompt_builder.build(
            query=query,
            chunks=retrieved_chunks
        )

        # 5. LLM 生成回答
        llm_answer = await self.llm.generate(prompt)

        elapsed = round(time.time() - start_time, 3)

        return RAGResult(
            answer=llm_answer.content,
            sources=retrieved_chunks,
            evidence_chunks=evidence_chunks,  # 新增（模块11）
            metadata={
                "latency": elapsed,
                "top_k": self.top_k,
                "llm": llm_answer.metadata,
            }
        )
```



## 验收测试

```
src/test/test_answer_aggregator.py # 从这个模块开始，测试文件全写入test文件夹下
```



```
import asyncio
from pathlib import Path
from dotenv import load_dotenv


# ================== Path & Env ==================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / "LangSmith.env")

# ================== Loaders ==================

from src.loaders.md_loader import MarkdownLoader

# ================== Processors ==================

from src.processors.cleaner import TextCleaner
from src.processors.semantic.semantic_splitter import SemanticSplitter

# ================== Embedding / Store ==================

from src.embeddings.openai_embedding import OpenAIEmbedding
from src.vectorstores.memory_store import InMemoryVectorStore

# ================== Retriever / Reranker / Aggregator ==================

from src.retrievers.simple_retriever import SimpleRetriever
from src.rerankers.embedding_reranker import EmbeddingReranker
from src.aggregators.simple_aggregator import AnswerAggregator

# ================== Prompt / LLM / RAG ==================

from src.prompts.simple_prompt import SimplePromptBuilder
from src.llms.openai_chat import OpenAIChatLLM
from src.rag.rag_core import RAGCore


# ================== Chunk Builder ==================

cleaner = TextCleaner()
splitter = SemanticSplitter()


def load_md_chunks():
    loader = MarkdownLoader()
    docs = loader.load(BASE_DIR / "data" / "model10" / "test.md")

    all_chunks = []
    for doc in docs:
        doc.content = cleaner.clean(doc.content)
        all_chunks.extend(splitter.split(doc))

    print(f"[Semantic Splitter] Total chunks: {len(all_chunks)}")
    return all_chunks


# ================== RAG Test ==================

async def run_test(use_aggregator: bool):

    chunks = load_md_chunks()

    # Embedding
    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    # Vector store
    store = InMemoryVectorStore()
    store.add(embedded_chunks)

    # Retriever
    retriever = SimpleRetriever(embedder=embedder, vectorstore=store)

    # Prompt / LLM
    prompt_builder = SimplePromptBuilder()
    llm = OpenAIChatLLM()

    # Aggregator（关键差异）
    aggregator = (
        AnswerAggregator(
            max_chunks=3
        )
        if use_aggregator
        else None
    )

    rag = RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        reranker=EmbeddingReranker(),
        aggregator=aggregator,
        top_k=5,
    )

    result = await rag.answer(
        "在 RAG 系统中，Reranking 的作用是什么？"
    )

    # ================== Pretty Print ==================

    print("\n==============================")
    print("WITH Aggregator" if use_aggregator else "NO Aggregator")

    print("\n[Answer]")
    print(result.answer)

    print("\n[Evidence]")

    if use_aggregator:
        for e in result.evidence_chunks:
            print("----")
            print(f"Rank: {e.rank}")
            print(f"Score: {e.score:.4f}")
            print(f"Source: {e.source}")
            print(f"Chunk ID: {e.chunk_id}")
            print("Content:")
            print(e.content)
    else:
        for s in result.sources:
            print("----")
            print(f"Score: {s.score:.4f}")
            print("Content:")
            print(s.chunk.content)
            print("Metadata:")
            print(s.chunk.metadata)


# ================== Main ==================

async def main():
    await run_test(use_aggregator=False)
    await run_test(use_aggregator=True)


if __name__ == "__main__":
    asyncio.run(main())
```



## 测试结果

### 不使用聚合

![model_11_2](md_img/model_11_2.png)



### 使用聚合

![model_11_3](md_img/model_11_3.png)



## 增强 Aggregator

### 目标

用“最终答案”作为新 query，对 evidence chunks 再做一次相似度重排



### 统一similarity

```
src/utils/similarity.py
```



```
import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
```



### 增强 Aggregator代码

```
src/aggregators/answer_aware_aggregator.py
```



```
from typing import List

from src.aggregators.base import BaseAnswerAggregator
from src.aggregators.schemas import EvidenceChunk
from src.vectorstores.search_result import SearchResult
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.llms.openai_chat import OpenAIChatLLM
from src.prompts.base import BasePromptBuilder
from src.utils.similarity import cosine_similarity


class AnswerAwareAggregator(BaseAnswerAggregator):
    """
    使用最终 Answer 作为 query，
    对 rerank 后的 chunks 再做一次 evidence 级别重排
    """

    def __init__(
        self,
        llm: OpenAIChatLLM,
        prompt_builder: BasePromptBuilder,
        embedder: OpenAIEmbedding,
        max_chunks: int = 3,
    ):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.embedder = embedder
        self.max_chunks = max_chunks

    async def aggregate(
        self,
        query: str,
        results: List[SearchResult],
    ) -> List[EvidenceChunk]:
        """
        注意：这里返回的是 EvidenceChunk，不再是 SearchResult
        """

        # 1. 先生成最终答案（这是“答案感知”的关键）
        prompt = self.prompt_builder.build(
            query=query,
            chunks=results[: self.max_chunks],
        )
        llm_answer = await self.llm.generate(prompt)
        answer_text = llm_answer.content

        # 2. 对 answer 做 embedding
        answer_embedding = await self.embedder.embed_texts([answer_text])
        answer_vector = answer_embedding[0].vector

        # 3. 对每个 chunk 计算 answer-aware score
        evidence_chunks: List[EvidenceChunk] = []

        for idx, r in enumerate(results):
            chunk_vector = r.chunk.vector
            score = cosine_similarity(answer_vector, chunk_vector)

            evidence_chunks.append(
                EvidenceChunk(
                    rank=0,  # 稍后统一排序后再填
                    score=score,
                    content=r.chunk.content,
                    source=r.chunk.metadata.get("source"),
                    chunk_id=r.chunk.metadata.get("chunk_id"),
                )
            )

        # 4. 按 answer-aware score 重排
        evidence_chunks.sort(key=lambda x: x.score, reverse=True)

        # 5. 重新标 rank，并截断
        final_chunks = []
        for i, e in enumerate(evidence_chunks[: self.max_chunks]):
            e.rank = i + 1
            final_chunks.append(e)

        return final_chunks
```



### SimpleAggregator 在做什么

```
rerank 后结果
   ↓
选择 top-k
   ↓
保留 rank / score / source
```

特点：

- 不依赖 LLM
- 不引入新的 embedding
- **确定性强、便宜、稳定**
- 非常适合：
  - Debug
  - Benchmark
  - 小模型 / 离线任务



### Answer-aware Aggregator 在做什么

```
rerank 后 chunks
   ↓
生成答案
   ↓
用“答案”反过来重排 evidence
```

这是：

- 二阶段推理
- 有 feedback loop
- 引入新的不确定性 & 成本
- 但解释性、对齐度极强



### 实际使用建议（非常重要）

| 场景             | 用哪个                  |
| ---------------- | ----------------------- |
| 单元测试         | `SimpleAggregator`      |
| 对抗测试         | `SimpleAggregator`      |
| Reranker 调参    | `SimpleAggregator`      |
| Debug chunk 顺序 | `SimpleAggregator`      |
| Demo / UI 展示   | `AnswerAwareAggregator` |
| 高质量回答       | `AnswerAwareAggregator` |

-  **Simple 是“地基”**
-  **Answer-aware 是“外挂能力”**



### 测试代码

```
src/test/test_answer_aggregator.py
```



```
# 通过注释切换Aggregator

# Aggregator（关键差异）
    aggregator = (
        # AnswerAggregator(
        #     max_chunks=3
        # )
        AnswerAwareAggregator(
            llm=llm,
            prompt_builder=prompt_builder,
            embedder=embedder,
            max_chunks=3,
        )
        if use_aggregator
        else None
    )
```



### 验收测试

![model_11_4](md_img/model_11_4.png)



# 模块12：多文档 / 长文档答案聚合

## 模块定位

当查询涉及多文档时，生成一个 **统一答案**，并标注每条 evidence 来源。

**重点**：

- 保留 answer-aware 排序逻辑。
- 支持跨文档合并 evidence。
- 解决 chunk 太小导致语义割裂的问题。



## 输入

```
query: 用户问题（string）
results: 多文档检索得到的 SearchResult 列表（包含 chunk.vector 和 metadata）
```



## 输出（与模块11相同）

```
class AggregatedAnswer:
    answer: str
    evidence_chunks: List[EvidenceChunk]

class EvidenceChunk:
    rank: int
    score: float
    content: str
    source: Any
    chunk_id: int
```



## 项目结构

```
src/
└── aggregators/
    └── answer_aware_aggregator.py  # 修改为支持多文档聚合

└── prompts/
    ├── failure_aware_prompt_builder.py
```



![model_12_1](md_img/model_12_1.png)



## 核心实现

```
src/aggregators/answer_multitude_aggregator.py
```



```
from typing import List
import math

from src.aggregators.base import BaseAnswerAggregator
from src.aggregators.schemas import EvidenceChunk, AggregatedAnswer
from src.vectorstores.search_result import SearchResult
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.llms.openai_chat import OpenAIChatLLM
from src.prompts.base import BasePromptBuilder
from src.utils.similarity import cosine_similarity


class AnswerAwareAggregator(BaseAnswerAggregator):
    """
    跨文档 Answer-aware Evidence Aggregator

    特点：
    - 使用【所有 chunks】生成一个全局答案
    - 用答案反向评估每个 chunk 的有效性
    - 显式惩罚 keyword-bomb / 空洞段落
    """

    def __init__(
            self,
            llm: OpenAIChatLLM,
            prompt_builder: BasePromptBuilder,
            embedder: OpenAIEmbedding,
            max_chunks: int = 3,
    ):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.embedder = embedder
        self.max_chunks = max_chunks

    async def aggregate(
            self,
            query: str,
            results: List[SearchResult],
    ) -> AggregatedAnswer:

        # ==========================================================
        # Phase 1: 【真正的answer-aware】分析
        # 先分析各chunk对答案的贡献，再生成答案
        # ==========================================================

        # 对每个chunk进行独立分析
        chunk_analyses = []
        for r in results:
            # 构建单个chunk分析prompt
            single_chunk_prompt = f"""
            请分析以下文本片段对于回答问题"{query}"的相关性和重要性：

            文本内容：{r.chunk.content}

            请提供：
            1. 相关性评分 (0-10)
            2. 信息独特性 (是否提供了其他片段没有的关键信息)
            3. 信息可靠性 (内容是否可信且有依据)
            4. 简要分析说明
            """

            analysis = await self.llm.generate(single_chunk_prompt)
            chunk_analyses.append({
                'chunk': r.chunk,
                'result': r,
                'analysis': analysis.content
            })

        # 基于所有分析结果生成全局答案
        synthesis_prompt = f"""
        请基于以下多个信息源的分析结果，回答问题："{query}"

        信息源分析结果：
        """

        for i, item in enumerate(chunk_analyses):
            synthesis_prompt += f"\n--- 信息源 {i + 1} ---\n"
            synthesis_prompt += f"内容: {item['chunk'].content}\n"
            synthesis_prompt += f"分析: {item['analysis']}\n"

        synthesis_prompt += f"""
        请综合以上所有信息源，生成一个全面、准确且有证据支持的答案。
        注意解决可能存在的信息冲突，并突出最可靠的信息来源。
        """

        synthesized_answer = await self.llm.generate(synthesis_prompt)
        synthesized_text = synthesized_answer.content

        # ==========================================================
        # Phase 2: 计算 embedding
        # ==========================================================

        embeddings = await self.embedder.embed_texts(
            [query, synthesized_text]
        )
        query_vec = embeddings[0].vector
        answer_vec = embeddings[1].vector

        # ✅ 新增：所有 chunk embedding
        chunk_vecs = [r.chunk.vector for r in results]

        # ==========================================================
        # Phase 3: 跨文档 Evidence 评分（关键改动）
        # ==========================================================

        evidence_chunks: List[EvidenceChunk] = []

        raw_consensus = [
            cosine_similarity(answer_vec, v) for v in chunk_vecs
        ]
        max_c = max(raw_consensus) or 1.0
        consensus_scores = [c / max_c for c in raw_consensus]

        for idx, r in enumerate(results):
            chunk_vec = r.chunk.vector
            content = r.chunk.content

            # 1️⃣ 与最终答案的一致性（局部）
            answer_sim = cosine_similarity(answer_vec, chunk_vec)

            # 2️⃣ 与原始问题的相关性
            query_sim = cosine_similarity(query_vec, chunk_vec)

            # 3️⃣ 信息密度惩罚（keyword bomb 杀手）
            tokens = content.split()
            token_count = max(len(tokens), 1)
            unique_ratio = len(set(tokens)) / token_count
            info_density = math.log(1 + token_count) * unique_ratio

            # ======================================================
            # ✅ 4️⃣ 跨文档"共识奖励"（真正拉开差距的地方）
            # ======================================================

            consensus_boost = consensus_scores[idx]

            final_score = (
                    0.45 * answer_sim +
                    0.25 * query_sim +
                    0.15 * info_density +
                    0.15 * consensus_boost  # 🔥 新增项
            )

            evidence_chunks.append(
                EvidenceChunk(
                    rank=0,
                    score=final_score,
                    content=content,
                    source=r.chunk.metadata.get("source"),
                    chunk_id=r.chunk.metadata.get("chunk_id"),
                )
            )

        # ==========================================================
        # Phase 4: 排序 & 截断
        # ==========================================================

        evidence_chunks.sort(key=lambda x: x.score, reverse=True)

        final_chunks = []
        for i, e in enumerate(evidence_chunks[: self.max_chunks]):
            e.rank = i + 1
            final_chunks.append(e)

        return AggregatedAnswer(
            answer=synthesized_text,  # 经过跨文档分析的全局答案
            evidence_chunks=final_chunks
        )
```



## 提示词构造新增

```
src/prompts/failure_aware_prompt_builder.py
```



```
from typing import List
from src.prompts.base import BasePromptBuilder
from src.vectorstores.search_result import SearchResult


class FailureAwarePromptBuilder(BasePromptBuilder):
    """
    基于检索 + 聚合 + 推理的失败机制识别
    """

    def build(
        self,
        query: str,
        chunks: List[SearchResult],
    ) -> str:

        context = "\n\n".join(
            f"[Chunk {i + 1} | Source: {r.chunk.metadata.get('source')}] \n"
            f"{r.chunk.content}"
            for i, r in enumerate(chunks)
        )

        prompt=f"""
            你是一个严谨的问答系统，只能基于给定的上下文回答问题。
    
            == == == == == == == == == ==
            【上下文信息】
            == == == == == == == == == ==
            {context}
    
            == == == == == == == == == ==
            【问题】
            == == == == == == == == == ==
            {query}
    
            == == == == == == == == == ==
            【回答要求（必须遵守）】
            == == == == == == == == == ==
    
            - 你的目标是给出 ** 问题的最终答案 **
            - 只有在上下文 ** 明确、一致地支持某个结论 ** 时，才能给出确定性回答
            - 如果上下文中存在：
            - 信息不完整
            - 关键证据缺失
            - 多个片段之间结论不一致
            请在答案中明确反映这种不确定性（例如使用条件化或范围性表述）
    
            - 不允许引入上下文之外的常识或推测
            - 不需要解释推理过程
            - 不需要评价上下文质量
            - 不需要比较不同方法
    
            == == == == == == == == == ==
            【输出格式】
            == == == == == == == == == ==
    
            直接给出问题的答案，不要包含分析说明。
        """.strip()

        return prompt
```



## RAGCore修改

```
src/rag/rag_core.py
```



```
# Aggregator 返回 AggregatedAnswer
aggregated = None
if self.aggregator:
    aggregated = await self.aggregator.aggregate(
        query=query,
        results=retrieved_chunks
    )

# LLM Answer 的“来源”变清晰
if aggregated:
    final_answer = aggregated.answer
    evidence_chunks = aggregated.evidence_chunks
else:
    final_answer = llm_answer.content
    evidence_chunks = []

# 修改后的关键 return
return RAGResult(
    answer=final_answer,
    sources=retrieved_chunks,
    evidence_chunks=evidence_chunks,
    metadata={
        "latency": elapsed,
        "top_k": self.top_k,
        "llm": llm_answer.metadata,
    }
)
```



## 验收测试

- **Embedding-only top-K**：直接用向量相似度排序，容易被高关键词段落欺骗。
- **Answer-aware**：先生成初步答案，然后基于答案 embedding 重新计算每个 chunk 的相关性，跨文档找出真正回答问题的段落。



```
src/test/test_multidoc_comparison.py
```



```
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from src.aggregators.schemas import AggregatedAnswer
from src.loaders.md_loader import MarkdownLoader
from src.processors.cleaner import TextCleaner
from src.processors.semantic.semantic_splitter import SemanticSplitter

from src.embeddings.openai_embedding import OpenAIEmbedding
from src.prompts.failure_aware_prompt_builder import FailureAwarePromptBuilder
from src.vectorstores.memory_store import InMemoryVectorStore

from src.retrievers.simple_retriever import SimpleRetriever
from src.rerankers.embedding_reranker import EmbeddingReranker
from src.prompts.simple_prompt import SimplePromptBuilder
from src.llms.openai_chat import OpenAIChatLLM
from src.rag.rag_core import RAGCore

from src.aggregators.answer_multitude_aggregator import AnswerAwareAggregator

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / "LangSmith.env")

cleaner = TextCleaner()
splitter = SemanticSplitter()

def load_md_chunks_multi():
    loader = MarkdownLoader()
    folder = BASE_DIR / "data" / "model12"
    all_chunks = []

    for file_path in folder.glob("*.md"):  # 遍历所有 md 文件
        docs = loader.load(file_path)
        for doc in docs:
            doc.content = cleaner.clean(doc.content)
            all_chunks.extend(splitter.split(doc))

    print(f"[Semantic Splitter] Total chunks: {len(all_chunks)}")
    return all_chunks


async def run_comparison(query: str):
    chunks = load_md_chunks_multi()

    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    store = InMemoryVectorStore()
    store.add(embedded_chunks)

    retriever = SimpleRetriever(embedder=embedder, vectorstore=store)
    llm = OpenAIChatLLM()
    prompt_builder = FailureAwarePromptBuilder()

    # ------------------ Embedding-only RAG ------------------
    rag_embedding_only = RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        reranker=EmbeddingReranker(),
        aggregator=None,
        top_k=5
    )

    result_embedding = await rag_embedding_only.answer(query)

    print("\n==============================")
    print("[Embedding-only top-K]")
    print("\n[Answer]")
    print(result_embedding.answer)
    print("\n[Evidence]")
    for s in result_embedding.sources:
        print("----")
        print(f"Score: {s.score:.4f}")
        print(f"Source: {s.chunk.metadata.get('source')}")
        print(f"Chunk ID: {s.chunk.metadata.get('chunk_id')}")
        print("Content:")
        print(s.chunk.content)

    # ------------------ Answer-aware RAG ------------------
    prompt_builder = FailureAwarePromptBuilder()

    aggregator = AnswerAwareAggregator(
        llm=llm,
        prompt_builder=prompt_builder,
        embedder=embedder,
        max_chunks=3
    )

    rag_answer_aware = RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        reranker=EmbeddingReranker(),
        aggregator=aggregator,
        top_k=5
    )

    result_answer_aware = await rag_answer_aware.answer(query)

    print("\n==============================")
    print("[Answer-aware top-K]")
    print("\n[Answer]")
    print(result_answer_aware.answer)
    print("\n[Evidence]")

    for e in result_answer_aware.evidence_chunks:
        print("----")
        print(f"Rank: {e.rank}")
        print(f"Score: {e.score:.4f}")
        print(f"Source: {e.source}")
        print(f"Chunk ID: {e.chunk_id}")
        print("Content:")
        print(e.content)


if __name__ == "__main__":
    asyncio.run(run_comparison("根据文档6的详细解释，对比传统重排和answer-aware重排在对抗性环境下的表现差异"))
```



## 测试结果

### 不使用聚合器

![model_12_2](md_img/model_12_2.png)



### 使用聚合器

![model_12_3](md_img/model_12_3.png)



# 模块13：前端开发

## 模块定位

增加可视化，用户友好



## 输入

```
query: str
```



## 输出

```
{
  "answer": {
    "content": "### 1️⃣ 失败分析（核心）\n\n在对抗性文档中..."
  },
  "evidences": [
    {
      "rank": 1,
      "score": 0.9194,
      "source": "doc3.md",
      "chunk_id": 0,
      "content": "在检索增强生成系统中，模型通常会面对大量 chunks..."
    }
  ]
}
```



## 增加依赖（requirements.txt）：

```
# -----------------------------
# 模块 1：Document Loader 依赖
# -----------------------------
pdfplumber>=0.10.0

# -----------------------------
# 模块 2：TextProcessor / TextSplitter 依赖
# -----------------------------
# 纯 Python，无新增依赖（已包含在 Python 标准库）

# -----------------------------
# 模块 3：Embedding Model 依赖
# -----------------------------
numpy>=1.24
openai>=1.28.0

# -----------------------------
# Web / API
# -----------------------------
fastapi>=0.110
uvicorn>=0.27
pydantic>=2.5

# -----------------------------
# 可选 / 工程化建议
# -----------------------------
# python-dotenv 可选，用于管理 OPENAI_API_KEY
python-dotenv>=1.0.0
# LangChain / LangSmith（实验 & 观测层）
langchain>=0.1.16
langchain-openai>=0.0.6
langsmith>=0.1.0
```



## 控制台运行指令

```
pip install -r requirements.txt
```



## 项目结构

```
# 和 src/ 同级
app/
	├─ main.py          # FastAPI + HTML 服务
	└──rag_app.py       # RAG 应用工厂
	
	├─ templates/
 	   └─ index.html    # 前端页面
```

![model_13_1](md_img/model_13_1.png)



## RAG应用开发

```
app/rag_app.py
```



```
from pathlib import Path
from dotenv import load_dotenv

from src.loaders.md_loader import MarkdownLoader
from src.processors.cleaner import TextCleaner
from src.processors.semantic.semantic_splitter import SemanticSplitter

from src.embeddings.openai_embedding import OpenAIEmbedding
from src.prompts.failure_aware_prompt_builder import FailureAwarePromptBuilder
from src.vectorstores.memory_store import InMemoryVectorStore

from src.retrievers.simple_retriever import SimpleRetriever
from src.rerankers.embedding_reranker import EmbeddingReranker
from src.llms.openai_chat import OpenAIChatLLM
from src.rag.rag_core import RAGCore
from src.aggregators.answer_multitude_aggregator import AnswerAwareAggregator

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "LangSmith.env")


def build_rag_core() -> RAGCore:
    # ---------- 文档加载 & 预处理 ----------
    cleaner = TextCleaner()
    splitter = SemanticSplitter()
    loader = MarkdownLoader()

    folder = BASE_DIR / "data" / "model12"
    all_chunks = []

    for file_path in folder.glob("*.md"):
        docs = loader.load(file_path)
        for doc in docs:
            doc.content = cleaner.clean(doc.content)
            all_chunks.extend(splitter.split(doc))

    print(f"[Semantic Splitter] Total chunks: {len(all_chunks)}")

    # ---------- Embedding & VectorStore ----------
    embedder = OpenAIEmbedding()
    store = InMemoryVectorStore()

    # 这里是 async 的，但 FastAPI 启动时无法 await
    # 所以我们用一个 trick（见下）
    store._pending_chunks = all_chunks
    store._embedder = embedder

    # ---------- 组件 ----------
    retriever = SimpleRetriever(embedder=embedder, vectorstore=store)
    reranker = EmbeddingReranker()
    llm = OpenAIChatLLM()
    prompt_builder = FailureAwarePromptBuilder()

    aggregator = AnswerAwareAggregator(
        llm=llm,
        prompt_builder=prompt_builder,
        embedder=embedder,
        max_chunks=3
    )

    return RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        reranker=reranker,
        aggregator=aggregator,
        top_k=5
    )
```



## FastAPI开发

```
app/main.py 
```



```
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from pathlib import Path

from app.rag_app import build_rag_core


class AskRequest(BaseModel):
    query: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_core = build_rag_core()

    store = rag_core.vectorstore
    embedder = rag_core.embedder

    embedded_chunks = await embedder.embed(store._pending_chunks)
    store.add(embedded_chunks)

    app.state.rag_core = rag_core
    print("[Lifespan] RAG system ready")

    yield

    print("[Lifespan] shutdown")


app = FastAPI(
    title="Minimal RAG Backend",
    lifespan=lifespan
)


@app.post("/ask")
async def ask(req: AskRequest):
    rag_core = app.state.rag_core
    result = await rag_core.answer(req.query)

    return {
        "answer": result.answer,
        "evidence": [
            {
                "rank": e.rank,
                "score": round(e.score, 4),
                "source": e.source,
                "chunk_id": e.chunk_id,
                "content": e.content,
            }
            for e in result.evidence_chunks
        ],
        "metadata": result.metadata
    }

@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Welcome to RAG System</h1><p>Template file not found.</p>"
    except Exception as e:
        return f"<h1>Error loading page</h1><p>{str(e)}</p>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8090)
```



## html开发

```
app/templates/index.html
```



```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RAG Research Demo</title>

    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            background: #f5f6f8;
            margin: 0;
            padding: 24px;
            color: #111827;
        }

        .container {
            max-width: 980px;
            margin: auto;
        }

        h1 {
            font-size: 22px;
            margin-bottom: 6px;
        }

        .subtitle {
            font-size: 14px;
            color: #6b7280;
            margin-bottom: 20px;
        }

        textarea {
            width: 100%;
            height: 60px;
            padding: 10px;
            font-size: 14px;
            border-radius: 4px;
            border: 1px solid #d1d5db;
            resize: vertical;
            box-sizing: border-box;
        }

        button {
            margin-top: 10px;
            padding: 6px 18px;
            font-size: 14px;
            background: #1f2937;
            color: #ffffff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }

        button:disabled {
            background: #9ca3af;
            cursor: not-allowed;
        }

        .section {
            margin-top: 24px;
        }

        .section-title {
            font-size: 15px;
            font-weight: 600;
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 4px;
            margin-bottom: 10px;
        }

        .answer {
            background: #ffffff;
            padding: 14px;
            border-radius: 4px;
            border: 1px solid #e5e7eb;
            white-space: pre-wrap;
            line-height: 1.6;
        }

        .evidence {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-left: 4px solid #2563eb;
            border-radius: 4px;
            padding: 10px;
            margin-bottom: 10px;
        }

        .evidence-header {
            font-size: 13px;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 6px;
        }

        .evidence pre {
            margin: 0;
            font-size: 13px;
            white-space: pre-wrap;
            line-height: 1.5;
        }

        .status {
            font-size: 14px;
            color: #374151;
        }

        .error {
            color: #b91c1c;
        }
    </style>
</head>

<body>
<div class="container">

    <h1>RAG Research Demo</h1>
    <div class="subtitle">
        Answer-aware evidence aggregation · Minimal experimental UI
    </div>

    <textarea id="query" placeholder="Enter your research question..."></textarea>
    <br>
    <button id="runBtn" onclick="ask()">Run Query</button>

    <div id="result"></div>

</div>

<script>
    async function ask() {
        const query = document.getElementById("query").value;
        const btn = document.getElementById("runBtn");
        const resultDiv = document.getElementById("result");
    
        // ---------- UI: running ----------
        btn.disabled = true;
        btn.innerText = "Running...";
        resultDiv.innerHTML = `
            <div class="section">
                <div class="section-title">Status</div>
                <div class="answer status">⏳ Running RAG pipeline…</div>
            </div>
        `;
    
        try {
            const res = await fetch("/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query })
            });
    
            const data = await res.json();
    
            let html = `
                <div class="section">
                    <div class="section-title">Answer</div>
                    <div class="answer">${data.answer}</div>
                </div>
            `;
    
            if (data.evidence && data.evidence.length > 0) {
                html += `
                    <div class="section">
                        <div class="section-title">Evidence Chunks</div>
                `;
                data.evidence.forEach(ev => {
                    html += `
                        <div class="evidence">
                            <div class="evidence-header">
                                Rank ${ev.rank} · Score ${ev.score} · Chunk ${ev.chunk_id}
                            </div>
                            <pre>${ev.content}</pre>
                        </div>
                    `;
                });
                html += `</div>`;
            }
    
            resultDiv.innerHTML = html;
    
        } catch (err) {
            console.error(err);
            resultDiv.innerHTML = `
                <div class="section">
                    <div class="section-title">Error</div>
                    <div class="answer error">
                        ❌ Request failed. Check backend logs.
                    </div>
                </div>
            `;
        } finally {
            // ---------- UI: restore ----------
            btn.disabled = false;
            btn.innerText = "Run Query";
        }
    }
</script>

</body>
</html>
```



## 验收测试

### 前端输入问题

```
# 可自行修改，但尽量要根据data/model12下的内容进行提问
# 或者可自行添加文件进行测试（如果使用pdf，txt格式记得修改加载文件loader方式）

# 我这里就用模块12的测试问题
根据文档6的详细解释，对比传统重排和answer-aware重排在对抗性环境下的表现差异
```



## 测试结果

### 运行前

![model_13_2](md_img/model_13_2.png)



### 运行中

![model_13_3](md_img/model_13_3.png)



### 运行后

![model_13_4](md_img/model_13_4.png)



# 模块14：前端流式显示

## 模块定位

前端边生成边显示答案



## OpenAI Chat新增

```
src/llms/openai_chat.py
```



```
# --------------------- 新增 ---------------------
async def stream_generate(self, prompt: str):
    """
    async generator: 边生成边 yield chunk
    """
    stream = await self.client.chat.completions.create(
        model=self.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=self.temperature,
        stream=True  # 启用流式响应
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
        	yield chunk.choices[0].delta.content
```



## RAGCore新增

```
src/rag/rag_core.py
```



```
@traceable(name="rag_answer_stream")
    async def stream_answer(self, query: str):
        """
        返回 async generator，每次 yield 增量文本
        """
        # 1. 检索相关 chunks
        retrieved_chunks = await self.retriever.retrieve(
            query=query,
            top_k=self.top_k * 2
        )

        # 2. Rerank
        if self.reranker:
            query_embedding = await self.retriever.embedder.embed_texts([query])
            query_vector = query_embedding[0].vector

            retrieved_chunks = self.reranker.rerank(
                query_vector=query_vector,
                results=retrieved_chunks,
                top_k=self.top_k
            )

        # 3. Aggregator
        aggregated = None
        if self.aggregator:
            aggregated = await self.aggregator.aggregate(
                query=query,
                results=retrieved_chunks
            )

        # 4. 根据是否有聚合器决定生成方式
        if aggregated:
            # 流式返回聚合器生成的答案
            answer_text = aggregated.answer
            # 按字符流式返回
            for char in answer_text:
                yield char
                await asyncio.sleep(0.03)
        else:
            # 如果没有聚合器，使用原来流式生成方式
            # 构建 Prompt
            prompt = self.prompt_builder.build(
                query=query,
                chunks=retrieved_chunks
            )

            # LLM 流式生成
            async for chunk in self.llm.stream_generate(prompt):
                yield chunk  # 逐步返回文本

        # ⚠️ 注意：这里不返回最终 RAGResult，如果前端需要 Evidence / metadata，需要再调用 answer()
```



## RAG应用新增

```
app/rag_app.py
```



```
@app.get("/ask/stream")
async def ask_stream(query: str):
    """
    使用 SSE 流式返回 answer + 最终 evidence/metadata
    """
    rag_core = app.state.rag_core

    async def event_generator():
        # 逐步生成 answer
        async for chunk in rag_core.stream_answer(query):
            yield f"data: {json.dumps({'answer_chunk': chunk})}\n\n"

        # 最终获取 Evidence / metadata
        result = await rag_core.answer(query)
        
        # 构建证据列表，避免在f-string中使用多行表达式
        evidence_list = []
        for e in result.evidence_chunks:
            evidence_item = {
                'rank': e.rank, 
                'score': round(e.score, 4), 
                'source': str(e.source), 
                'chunk_id': e.chunk_id, 
                'content': e.content
            }
            evidence_list.append(evidence_item)
        
        # 处理 metadata 中可能包含的不可序列化对象
        metadata_serializable = {}
        for key, value in result.metadata.items():
            if isinstance(value, (Path, type(None))) or hasattr(value, '__fspath__'):
                metadata_serializable[key] = str(value)
            else:
                metadata_serializable[key] = value
        
        final_data = {
            'status': 'Done',
            'answer': result.answer,
            'evidence': evidence_list,
            'metadata': metadata_serializable
        }
        
        yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```



## html新增

```
app/templates/index.html
```



```
# 前面button的onclick方法改为askStream()
async function askStream() {
    const query = document.getElementById("query").value;
    const btn = document.getElementById("runBtn");
    const resultDiv = document.getElementById("result");

    btn.disabled = true;
    btn.innerText = "Running…";
    resultDiv.innerHTML = `<div class="answer" id="answer">⏳ Generating…</div>`;

    const answerDiv = document.getElementById("answer");

    const evtSource = new EventSource(`/ask/stream?query=${encodeURIComponent(query)}`);
    
    var count = 0;

    evtSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if(count === 0){
            answerDiv.textContent = "";
            count++;
        }

        if (data.answer_chunk) {
            answerDiv.textContent += data.answer_chunk;
        }

        if (data.status === 'Done') {
            btn.disabled = false;
            btn.innerText = "Run Query";
            
            // 渲染最终 evidence
            if (data.evidence && data.evidence.length > 0) {
                let html = "<h3>Evidence Chunks:</h3>";
                data.evidence.forEach(ev => {
                    html += `
                    <div class="evidence">
                        <strong>Rank ${ev.rank} | Score ${ev.score} | Source: ${ev.source} | Chunk ${ev.chunk_id}</strong>
                        <pre>${ev.content}</pre>
                    </div>`;
                });
                resultDiv.innerHTML += html;
            }
            
            evtSource.close();
        }
    };

    evtSource.onerror = () => {
        answerDiv.textContent = "❌ Stream failed";
        btn.disabled = false;
        btn.innerText = "Run Query";
        evtSource.close();
    };
}
```



## 测试结果

### 运行中

![model_14_1](md_img/model_14_1.png)



### 运行后

![model_14_2](md_img/model_14_2.png)



# 模块15：多轮对话

## 模块定位

开发多轮对话，增加系统可用性。



为什么要LLM 先 rewrite query，再进 RAG

1. 解决多轮对话的“指代问题”

​	**rewrite 的目标**：把当前轮 + 历史上下文 → 还原成一个“单轮、信息完整、可检索”的 query

2. 降低 embedding-only 检索的噪声
3. 为后续 rerank / answer-aware 铺路

​	answer-aware 再强，也建立在“检索没太歪”的前提上。所以rewrite 的价值是：把 query 拉回“文档空间”，避免一开始就检错方向



Rewrite Prompt 的核心原则

- **消解指代**
- **补全省略信息**
- **保持问题意图不变**



## 输入

```
session_id（对话上下文标识）
user_input（当前用户输入）
```



## 输出

```
RAGResult(
    answer: str,
    sources: List[SearchResult],
    evidence_chunks: List[EvidenceChunk],
    metadata: Dict
)
```



## 项目结构

```
src/
├─ chat/
   ├─ _init_.py
   ├─ rewriter.py       # QueryRewriter 核心模块
   ├─ session.py        # 会话状态
   ├─ memory.py         # 对话历史存储
   ├─ schemas.py        # ChatMessage / ChatTurn
   └─ service.py        # ChatService（核心）
```



![model_15_1](md_img/model_15_1.png)



## 对话数据结构

```
src/chat/schemas.py
```



```
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ChatMessage:
    role: str          # "user" | "assistant"
    content: str

@dataclass
class Evidence:
    rank: int
    score: float
    source: str
    chunk_id: str
    content: str

@dataclass
class ChatTurn:
    user_message: ChatMessage
    assistant_message: ChatMessage
    rewritten_query: str
    evidence: List[Evidence]
    metadata: Optional[dict] = None
```



## session 的结构

```
src/chat/session.py
```



```
from dataclasses import dataclass, field
from typing import Dict, Any
import time
import uuid


@dataclass
class ChatSession:
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new() -> "ChatSession":
        return ChatSession(session_id=str(uuid.uuid4()))

    def touch(self):
        self.updated_at = time.time()
```



## 会话内存

```
src/chat/memory.py
```



```
from collections import defaultdict
from typing import List
from .schemas import ChatMessage

class ChatMemory:

    def __init__(self, max_turns: int = 10):
        self.store = defaultdict(list)
        self.max_turns = max_turns

    def append(self, session_id: str, message: ChatMessage):
        self.store[session_id].append(message)

        # 限制长度（防 prompt 爆炸）
        if len(self.store[session_id]) > self.max_turns * 2:
            self.store[session_id] = self.store[session_id][-self.max_turns * 2:]

    def get(self, session_id: str) -> List[ChatMessage]:
        return self.store.get(session_id, [])
```



## ChatService

```
src/chat/service.py
```



```
from typing import List

from .memory import ChatMemory
from .schemas import ChatMessage, ChatTurn, Evidence
from .rewriter import BaseQueryRewriter
from src.rag.rag_core import RAGCore
from ..llms.openai_chat import OpenAIChatLLM


class LLMQueryRewriter(BaseQueryRewriter):

    def __init__(self, llm: OpenAIChatLLM, max_turns: int = 5):
        self.llm = llm
        self.max_turns = max_turns

    async def rewrite(self, query: str, history: List[ChatMessage]) -> str:
        history = history[-self.max_turns * 2:]

        history_text = "\n".join(
            f"{m.role}: {m.content}" for m in history
        )

        prompt = f"""
            你是一个多轮 RAG 系统中的 Query Rewriter。

            系统目标：
            - 用户的问题是递进的
            - 你需要基于“对话历史”，判断当前问题处于什么阶段
            - 只生成【检索新增信息】所需的 query
            - 避免重复检索已解释的基础概念
            
            对话历史（按时间顺序）：
            {history_text}
            
            当前用户问题：
            {query}
            
            请遵循以下规则生成检索 query：
            1. 如果是第一轮：生成“定义 / 原理”型检索 query
            2. 如果是追问失败 / 局限：生成“失败机制 / 对抗性情况”型检索 query
            3. 如果是方法对比：生成“对比 / 解决机制”型检索 query
            4. 不要复述用户问题
            5. 不要给出答案，只输出一条检索 query
            
            输出格式（严格）：
            <REWRITTEN_QUERY>
        """

        result = await self.llm.generate(prompt)
        rewritten = result.content.strip()

        # 防御：空 rewrite 就 fallback
        if not rewritten:
            return query

        return rewritten

class ChatService:

    def __init__(
        self,
        rag_core: RAGCore,
        rewriter: BaseQueryRewriter,
        memory: ChatMemory
    ):
        self.rag_core = rag_core
        self.rewriter = rewriter
        self.memory = memory

    async def chat(self, session_id: str, user_input: str) -> ChatTurn:
        # 1️⃣ 取历史
        history = self.memory.get(session_id)

        # 2️⃣ rewrite（多轮 → 单轮）
        rewritten_query = await self.rewriter.rewrite(user_input, history)

        # 3️⃣ 记录用户输入
        user_msg = ChatMessage(role="user", content=user_input)
        self.memory.append(session_id, user_msg)

        # 4️⃣ 核心：直接用你现在的 RAGCore
        rag_result = await self.rag_core.answer(rewritten_query)

        # 5️⃣ 记录 assistant
        assistant_msg = ChatMessage(
            role="assistant",
            content=rag_result.answer
        )
        self.memory.append(session_id, assistant_msg)

        # 6️⃣ Evidence 结构化（支持 embedding-only / answer-aware）
        evidence = []
        for e in rag_result.evidence_chunks:
            evidence.append(
                Evidence(
                    rank=e.rank,
                    score=e.score,
                    source=e.source,
                    chunk_id=e.chunk_id,
                    content=e.content
                )
            )

        return ChatTurn(
            user_message=user_msg,
            assistant_message=assistant_msg,
            rewritten_query=rewritten_query,
            evidence=evidence,
            metadata=rag_result.metadata
        )
```



## 验收测试

### rewrite是什么样子

```
Turn1
embedding-only rerank 的定义和工作原理

Turn2
embedding-only rerank 在对抗性文档或噪声干扰下失败的具体机制

Turn3
answer-aware rerank 如何通过生成答案反向评估证据以解决 embedding-only 的失败
```



```
src/test/test_chat_multiturn.py
```



```
import asyncio
import uuid
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / "LangSmith.env")

from src.loaders.md_loader import MarkdownLoader
from src.processors.cleaner import TextCleaner
from src.processors.semantic.semantic_splitter import SemanticSplitter

from src.embeddings.openai_embedding import OpenAIEmbedding
from src.prompts.failure_aware_prompt_builder import FailureAwarePromptBuilder
from src.vectorstores.memory_store import InMemoryVectorStore
from src.retrievers.simple_retriever import SimpleRetriever
from src.rerankers.embedding_reranker import EmbeddingReranker
from src.llms.openai_chat import OpenAIChatLLM
from src.aggregators.answer_multitude_aggregator import AnswerAwareAggregator
from src.rag.rag_core import RAGCore

from src.chat.service import ChatService, LLMQueryRewriter
from src.chat.memory import ChatMemory


cleaner = TextCleaner()
splitter = SemanticSplitter()


def load_md_chunks():
    loader = MarkdownLoader()
    folder = BASE_DIR / "data" / "model12"
    all_chunks = []

    for file_path in folder.glob("*.md"):
        docs = loader.load(file_path)
        for doc in docs:
            doc.content = cleaner.clean(doc.content)
            all_chunks.extend(splitter.split(doc))

    print(f"[Loader] total chunks = {len(all_chunks)}")
    return all_chunks


async def build_chat_service():
    #  文档 & 向量
    chunks = load_md_chunks()

    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    store = InMemoryVectorStore()
    store.add(embedded_chunks)

    retriever = SimpleRetriever(embedder=embedder, vectorstore=store)
    llm = OpenAIChatLLM()
    prompt_builder = FailureAwarePromptBuilder()

    aggregator = AnswerAwareAggregator(
        llm=llm,
        prompt_builder=prompt_builder,
        embedder=embedder,
        max_chunks=3
    )

    rag_core = RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        reranker=EmbeddingReranker(),
        aggregator=aggregator,
        top_k=5
    )

    memory = ChatMemory(max_turns=10)
    rewriter = LLMQueryRewriter(llm, max_turns=5)

    return ChatService(rag_core=rag_core, memory=memory, rewriter=rewriter)


async def run_multiturn_chat():
    chat_service = await build_chat_service()
    session_id = str(uuid.uuid4())

    questions = [
        "什么是 embedding-only rerank？",
        "它在什么情况下会失败？",
        "那 answer-aware 方法是怎么解决这个问题的？"
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n================ Turn {i} ================")
        print("[User]")
        print(q)

        result = await chat_service.chat(session_id, q)

        print("\n[Assistant]")
        print(result.assistant_message.content)

        if result.evidence:
            print("\n[Evidence]")
            for e in result.evidence:
                print(f"- Rank {e.rank} | Score {e.score:.4f} | Chunk {e.chunk_id}")


if __name__ == "__main__":
    asyncio.run(run_multiturn_chat())
```



## 测试结果

### 第一轮

![model_15_2](md_img/model_15_2.png)



### 第二轮

![model_15_3](md_img/model_15_3.png)



### 第三轮

![model_15_4](md_img/model_15_4.png)



# 模块16：证据支持度Reranker

## 模块定位

**模块9的reranker满足了：**

- 输入清晰（query_vector + chunk vectors）

- 输出稳定（排序后的 SearchResult）

- 可复用、可组合

- 适合做 第一阶段 rerank

**但信息来源只有一个：**query embedding



**只能回答这一类问题**：在“语义相似”的候选里，谁更像 query？



**无法回答**

- 这个 chunk 有没有 **因果解释**
- 有没有 **证据**
- 是不是 **真正回答了 why / how**
- 是否只是“关键词对上了”



**模块16目标**

- 继承 `BaseReranker`（体系一致）
- 可以和 `EmbeddingReranker` 串联使用
- 不负责最终回答，只负责“证据判断”
- 输入里必须显式出现「draft answer」
- 输出仍然是 `List[SearchResult]`



## 输入

```
query: str,
draft_answer: str,
results: List[SearchResult],
top_k: int = 5
```



## 输出

```
List[SearchResult]
```



## 项目结构

```
src/
├── rerankers/
    ├── __init__.py
    ├── base.py
    ├── embedding_reranker.py
    └── answer_aware_reranker.py   # 新增
```



![model_16_1](md_img/model_16_1.png)



## 证据支持度Reranker

```
src/rerankers/answer_aware_reranker.py
```



```
from typing import List
import numpy as np

from src.rerankers.base import BaseReranker
from src.vectorstores.search_result import SearchResult


class AnswerAwareReranker(BaseReranker):
    """
    Answer-aware reranker:
    使用 LLM 判断 chunk 是否真正支持 draft answer
    """

    def __init__(
        self,
        llm,
        query_text: str,
        draft_answer: str
    ):
        self.llm = llm
        self.query_text = query_text
        self.draft_answer = draft_answer

    async def rerank(
        self,
        query_vector: np.ndarray,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:

        if not results:
            return []

        reranked = []

        for r in results:
            score = await self._score_chunk(r.chunk.content)

            reranked.append(
                SearchResult(
                    chunk=r.chunk,
                    score=score
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]

    async def _score_chunk(self, chunk_text: str) -> float:
        prompt = f"""
            You are evaluating whether a document chunk SUPPORTS a draft answer.

            Important:
            - Mere topical similarity or definition overlap is NOT support.
            - The chunk must provide:
              (a) causal explanation, OR
              (b) factual evidence, OR
              (c) reasoning that directly justifies the draft answer.
            
            [Question]
            {self.query_text}
            
            [Draft Answer]
            {self.draft_answer}
            
            [Document Chunk]
            {chunk_text}
            
            Task:
            Does this chunk provide concrete support for the draft answer?
            
            Scoring:
            - 1.0 = explicitly supports the answer with reasoning or evidence
            - 0.5 = partially relevant but weak or indirect support
            - 0.0 = only topically related or unrelated
            
            Only output a number between 0.0 and 1.0.

        """.strip()

        response = await self.llm.generate(prompt)


        try:
            return float(response.content.strip())
        except Exception:
            return 0.0
```



## RagCore新增

```
src/rag/rag_core.py
```



```
初始化新增
    def __init__(
        self,
        embedder: BaseEmbedding,
        vectorstore: BaseVectorStore,
        retriever: BaseRetriever,
        prompt_builder: BasePromptBuilder,
        llm: BaseLLM,
        reranker=None, 
        aggregator=None, 
        top_k: int = 5,
        enable_answer_aware: bool = False # 新增（可选）
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.reranker = reranker 
        self.aggregator = aggregator 
        self.top_k = top_k
        self.enable_answer_aware = enable_answer_aware # 新增（可选）

新增细排方法
# 3. Answer-aware rerank（细排）
        if self.enable_answer_aware:
            draft_prompt = self.prompt_builder.build(
                query=query,
                chunks=retrieved_chunks
            )

            draft_answer = await self.llm.generate(draft_prompt)

            answer_reranker=AnswerAwareReranker(
                llm=self.llm,
                query_text=query,
                draft_answer=draft_answer.content
            )

            print(f"Before answer-aware rerank: {[r.score for r in retrieved_chunks[:3]]}")

            retrieved_chunks = await answer_reranker.rerank(
                query_vector=query_vector,
                results=retrieved_chunks,
                top_k=self.top_k,
            )

            print(f"After answer-aware rerank: {[r.score for r in retrieved_chunks[:3]]}")
```



## 验收测试

```
src/test/test_chat_multiturn.py
```



```
rag_core = RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        reranker=EmbeddingReranker(),
        aggregator=aggregator,
        top_k=5,
        enable_answer_aware=True	# 插入新reranker
    )
```



## 测试结果

### 结果说明

现在有3个score，分别如下

| 阶段                | score 实际含义      |
| ------------------- | ------------------- |
| embedding rerank    | 查询相似度          |
| answer-aware rerank | 证据支持度          |
| aggregator          | 证据整体贡献 / 权重 |

**最后展示的得分是聚合器得分**（这也是为什么最后有的分数超过1的原因），所以我在新增细排方法里写了这两句，方便展示证据支持度的得分，这样可以对比3个阶段的得分

```
# Embedding-only rerank（相似度）
print(f"Before answer-aware rerank: {[r.score for r in retrieved_chunks[:3]]}")

# Answer-aware rerank（支持度）
print(f"After answer-aware rerank: {[r.score for r in retrieved_chunks[:3]]}")

# 最后看到的得分 Aggregator score（全局证据贡献）
```



### 第一轮

![model_16_2](md_img/model_16_2.png)



### 第二轮

![model_16_3](md_img/model_16_3.png)



### 第三轮

![model_16_4](md_img/model_16_4.png)



### 系统改进的量化证明

| 指标          | Before Answer-aware | After Answer-aware | 改进程度         |
| ------------- | ------------------- | ------------------ | ---------------- |
| 平均分数      | 0.51                | 0.89               | **+74.5%**       |
| 满分chunk比例 | 0%                  | 77.8% (7/9)        | **大幅提升**     |
| 零分chunk比例 | 0%                  | 22.2% (2/9)        | **有效过滤噪声** |



# 模块17：



