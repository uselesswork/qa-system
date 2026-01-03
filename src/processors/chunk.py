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
