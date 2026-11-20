from __future__ import annotations

import json
import re
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "is", "it", "to", "for", "of", "in", "on", "and", "or",
    "my", "i", "we", "you", "this", "that", "with", "be", "are", "was", "have",
}


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2 and t not in _STOPWORDS}


def retrieve_top_articles(query_text: str, max_k: int = 2, margin: int = 3) -> str:
    """Adaptive lexical retriever: sends top-1 when confidence margin is clear, else top-2."""
    query_tokens = _tokenize(query_text)
    scored: list[tuple[int, str, str]] = []
    for aid in article_ids():
        body = load_article(aid)
        id_overlap = len(query_tokens & _tokenize(aid.replace("-", " "))) * 3
        body_overlap = len(query_tokens & _tokenize(body))
        scored.append((id_overlap + body_overlap, aid, body.strip()))
    scored.sort(key=lambda x: (-x[0], x[1]))

    if len(scored) >= 2 and scored[0][0] >= scored[1][0] + margin:
        selected = scored[:1]
    else:
        selected = scored[:max_k]
    return "\n\n".join(f"[{aid}]\n{body}" for _, aid, body in selected)

@dataclass
class TicketResolution:
    action: str
    kb_article: str
    customer_reply: str
    raw_completion: str


def parse_resolution(text: str) -> TicketResolution:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON in model output:\n{text[:400]}")
    data = json.loads(m.group())
    return TicketResolution(
        action=str(data["action"]).upper(),
        kb_article=str(data["kb_article"]),
        customer_reply=str(data.get("customer_reply", "")),
        raw_completion=text,
    )


_SYSTEM_PROMPT = (
    "You are ShopAssist, a Tier-1 e-commerce support agent. "
    "Choose action (REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2), cite the matching kb_article, "
    "and write a brief customer_reply. Return ONLY valid JSON."
)


class SupportCopilot:
    """ShopAssist Tier-1 copilot (cost-optimized single call with local retrieval)."""

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def build_messages(self, ticket_id: str) -> list[ChatMessage]:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        snapshot = json.dumps(order, separators=(",", ":"))
        policies = retrieve_top_articles(f"{thread} {snapshot}", max_k=2)

        user = "\n\n".join(
            [
                f"TICKET_ID: {ticket_id}",
                f"POLICIES:\n{policies}",
                f"THREAD:\n{thread}",
                f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
            ]
        )
        return [ChatMessage("system", _SYSTEM_PROMPT), ChatMessage("user", user)]

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        messages = self.build_messages(ticket_id)
        completion = self.llm.chat("handle_ticket", messages)
        return parse_resolution(completion)