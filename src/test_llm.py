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
