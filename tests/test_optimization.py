import pytest

from eval.reference import decide_order
from question1.agent import SupportCopilot, retrieve_top_articles
from question2.agent import SupportPipeline
from shared.tools import get_order, get_ticket

PUBLIC_TICKETS = ["T-1001", "T-1002", "T-1003", "T-1004"]
ALL_TICKETS = PUBLIC_TICKETS + ["T-1091"]


def test_q1_accuracy_and_budget():
    for tid in ALL_TICKETS:
        copilot = SupportCopilot()
        res = copilot.handle_ticket(tid)
        expected = decide_order(get_order(get_ticket(tid)["order_id"]))
        assert res.action == expected.action, f"Q1 action mismatch on {tid}"
        assert res.kb_article == expected.kb_article, f"Q1 KB mismatch on {tid}"
        assert len(copilot.session.calls) == 1, f"Q1 should make 1 LLM call on {tid}"
        if tid in PUBLIC_TICKETS:
            pts = copilot.session.total_points()
            assert pts <= 3000, f"Q1 exceeded 3000-pt budget on {tid}: {pts}"


def test_q2_accuracy_and_budget():
    for tid in ALL_TICKETS:
        pipe = SupportPipeline()
        res = pipe.handle_ticket(tid)
        expected = decide_order(get_order(get_ticket(tid)["order_id"]))
        assert res.action == expected.action, f"Q2 action mismatch on {tid}"
        assert res.kb_article == expected.kb_article, f"Q2 KB mismatch on {tid}"
        assert len(pipe.session.calls) == 1, f"Q2 should make 1 LLM call on {tid}"
        assert pipe.session.calls[0].label == "compose"
        if tid in PUBLIC_TICKETS:
            pts = pipe.session.total_points()
            assert pts <= 2500, f"Q2 exceeded 2500-pt budget on {tid}: {pts}"


def test_adaptive_retrieval_returns_relevant_policy():
    out = retrieve_top_articles("digital ebook license already revealed refund")
    assert "digital-license-no-refund" in out


def test_q2_audit_rejects_invalid_action():
    pipe = SupportPipeline()
    bad_json = '{"action": "FREE_MONEY", "kb_article": "returns-within-30-days"}'
    with pytest.raises(ValueError, match="invalid action"):
        pipe._audit(bad_json)


def test_q2_audit_rejects_unknown_kb_article():
    pipe = SupportPipeline()
    bad_json = '{"action": "REFUND", "kb_article": "nonexistent-policy-999"}'
    with pytest.raises(ValueError, match="unknown kb_article"):
        pipe._audit(bad_json)