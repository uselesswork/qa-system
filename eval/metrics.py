"""离线评测指标；相关性以人工标注的文档来源名为准。"""

from __future__ import annotations

from statistics import mean, median
import re


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 3)


def evaluate(cases: list[dict], predictions: list[dict], k: int = 5) -> dict:
    if k < 1:
        raise ValueError("k 必须大于 0")
    case_ids = [case["id"] for case in cases]
    prediction_ids = [item["id"] for item in predictions]
    if len(set(case_ids)) != len(case_ids) or len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("样本或预测存在重复 ID")
    if set(case_ids) != set(prediction_ids):
        raise ValueError(f"样本与预测 ID 不一致：缺少 {sorted(set(case_ids)-set(prediction_ids))}，多出 {sorted(set(prediction_ids)-set(case_ids))}")
    by_id = {item["id"]: item for item in predictions}
    rows = []
    for case in cases:
        prediction = by_id[case["id"]]
        gold = set(case["relevant_sources"])
        retrieved = prediction["retrieved_sources"][:k]
        cited = prediction["cited_sources"]
        if not isinstance(retrieved, list) or not isinstance(cited, list):
            raise ValueError("retrieved_sources 和 cited_sources 必须是列表")
        reciprocal_rank = next((1 / rank for rank, source in enumerate(retrieved, 1)
                                if source in gold), 0.0)
        answer = prediction["answer"]
        normalized_answer = re.sub(r"\s+", "", answer)
        refused = bool(prediction["refused"])
        row = {
            "id": case["id"], "category": case["category"],
            "hit_at_k": bool(reciprocal_rank) if gold else None,
            "reciprocal_rank": reciprocal_rank if gold else None,
            "citation_precision": (sum(source in gold for source in cited) / len(cited))
            if cited and gold else (0.0 if gold else None),
            "refusal_correct": refused if case["should_refuse"] else None,
            "answer_contains_all": (
                all(re.sub(r"\s+", "", term) in normalized_answer
                    for term in case["answer_contains"])
                and not any(re.sub(r"\s+", "", term) in normalized_answer
                            for term in case.get("answer_excludes", []))
            )
            if not case["should_refuse"] else None,
            "latency_ms": float(prediction["latency_ms"]),
            "model_calls": int(prediction["model_calls"]),
        }
        if row["latency_ms"] < 0 or row["model_calls"] < 0:
            raise ValueError("耗时和调用次数不能为负")
        rows.append(row)
    answerable = [row for row in rows if row["hit_at_k"] is not None]
    refusal = [row for row in rows if row["refusal_correct"] is not None]
    latencies = [row["latency_ms"] for row in rows]
    return {
        "sample_count": len(rows), "answerable_count": len(answerable),
        "refusal_count": len(refusal), "k": k,
        "hit_at_k": mean(row["hit_at_k"] for row in answerable) if answerable else None,
        "mrr_at_k": mean(row["reciprocal_rank"] for row in answerable) if answerable else None,
        "citation_precision": mean(row["citation_precision"] for row in answerable) if answerable else None,
        "refusal_accuracy": mean(row["refusal_correct"] for row in refusal) if refusal else None,
        "answer_contains_accuracy": mean(row["answer_contains_all"] for row in answerable) if answerable else None,
        "latency_p50_ms": round(median(latencies), 3) if latencies else None,
        "latency_p95_ms": percentile(latencies, .95),
        "model_calls_total": sum(row["model_calls"] for row in rows),
        "model_calls_mean": mean(row["model_calls"] for row in rows) if rows else None,
        "cases": rows,
    }
