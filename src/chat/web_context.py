"""会话内已保存的网页回答，仅作为该会话的可追溯上下文。"""

import json


def web_turns(messages: list[dict], limit: int = 3) -> list[dict]:
    turns = []
    for index, message in enumerate(messages):
        if (message.get("role") != "assistant"
                or message.get("metadata", {}).get("mode") not in ("web", "session_web_context")):
            continue
        sources = [item for item in message.get("evidence", [])
                   if item.get("source_type") == "web" and item.get("url")]
        if not sources:
            continue
        previous = messages[index - 1] if index else {}
        turns.append({"question": previous.get("content", "") if previous.get("role") == "user" else "",
                      "answer": message.get("content", ""), "sources": sources})
    return turns[-limit:]


async def answer_from_web_context(repository, llm, user_id: str, session_id: str,
                                  question: str) -> dict | None:
    turns = web_turns(repository.get(user_id, session_id)["messages"])
    if not turns:
        return None
    excerpts = "\n\n".join(
        f"记录 {index}：问题：{turn['question']}\n回答：{turn['answer']}"
        for index, turn in enumerate(turns, 1)
    )
    verdict = await llm.generate(f"""
判断当前问题能否只根据同一会话里已保存的联网回答来回答。
当前问题：{question}
{excerpts}
只有这些回答直接包含所需事实，或当前问题明确是对其中一项的追问时才通过。
不能使用模型常识。只返回 JSON：
{{"supported": false, "index": 0, "quote": ""}}
或 {{"supported": true, "index": 1, "quote": "对应回答中的连续原文摘录"}}。
""".strip())
    try:
        result = json.loads(verdict.content.strip())
        index = result["index"]
        quote = result["quote"]
        if (result.get("supported") is not True or type(index) is not int
                or not 1 <= index <= len(turns) or not isinstance(quote, str)
                or len(quote.strip()) < 4 or quote not in turns[index - 1]["answer"]):
            return None
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    turn = turns[index - 1]
    response = await llm.generate(f"""
只依据这条先前联网回答，回答当前问题。不得补充其中没有的事实。
先前问题：{turn['question']}
先前回答：{turn['answer']}
当前问题：{question}
如果资料仍不足，直接说资料不足。
""".strip())
    answer = response.content.strip()
    if not answer:
        return None
    evidence = [{**source, "rank": rank} for rank, source in enumerate(turn["sources"], 1)]
    metadata = {"mode": "session_web_context", "citations": [],
                "model": getattr(llm, "model", None)}
    repository.append_turn(user_id, session_id, question, answer, question, evidence, metadata)
    return {"session_id": session_id, "user_message": question, "answer": answer,
            "rewritten_query": question, "evidence": evidence, "metadata": metadata}
