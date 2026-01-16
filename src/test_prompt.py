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
