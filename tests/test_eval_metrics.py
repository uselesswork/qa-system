import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "eval" / "metrics.py"
SPEC = importlib.util.spec_from_file_location("qa_eval_metrics", MODULE_PATH)
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def case(id, gold, refuse=False):
    return {"id": id, "category": "测试", "relevant_sources": gold,
            "answer_contains": [] if refuse else ["正确"], "should_refuse": refuse}


def prediction(id, retrieved, cited, answer="正确", refused=False, latency=100):
    return {"id": id, "retrieved_sources": retrieved, "cited_sources": cited,
            "answer": answer, "refused": refused, "latency_ms": latency, "model_calls": 2}


def test_metrics_handle_ranking_noise_and_no_answer():
    report = metrics.evaluate(
        [case("one", ["a.md"]), case("two", ["b.md"]), case("none", [], True)],
        [prediction("two", ["noise.md", "b.md"], ["noise.md", "b.md"], latency=200),
         prediction("none", ["noise.md"], [], "资料不足", True, 300),
         prediction("one", ["a.md"], ["a.md"], latency=100)], k=2)
    assert report["hit_at_k"] == 1
    assert report["mrr_at_k"] == .75
    assert report["citation_precision"] == .75
    assert report["refusal_accuracy"] == 1
    assert report["answer_contains_accuracy"] == 1
    assert report["latency_p50_ms"] == 200
    assert report["latency_p95_ms"] == 290
    assert report["model_calls_total"] == 6


def test_metrics_reject_missing_predictions_and_bad_k():
    with pytest.raises(ValueError, match="ID 不一致"):
        metrics.evaluate([case("one", ["a.md"])], [], 5)
    with pytest.raises(ValueError, match="k 必须"):
        metrics.evaluate([], [], 0)


def test_answer_check_normalizes_spacing_and_rejects_obsolete_date():
    sample = case("date", ["project.md"])
    sample["answer_contains"] = ["2026年6月18日"]
    sample["answer_excludes"] = ["2026年6月15日"]
    good = prediction("date", ["project.md"], ["project.md"],
                      answer="正式日期：2026 年 6 月 18 日。")
    assert metrics.evaluate([sample], [good])["answer_contains_accuracy"] == 1
    bad = prediction("date", ["project.md"], ["project.md"],
                     answer="2026 年 6 月 18 日，旧草案为 2026 年 6 月 15 日。")
    assert metrics.evaluate([sample], [bad])["answer_contains_accuracy"] == 0
