from __future__ import annotations

import json
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import article_ids
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

from question1.agent import parse_resolution, retrieve_top_articles

_VALID_ACTIONS = {"REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2"}

_COMPOSE_SYSTEM = (
    "You are ShopAssist, a Tier-1 e-commerce support agent. "
    "Choose action (REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2), cite the matching kb_article, "
    "and write a brief customer_reply. Return ONLY valid JSON."
)


@dataclass
class TicketResolution:
    action: str
    kb_article: str


class SupportPipeline:
    """
    ShopAssist 4-stage pipeline: triage -> retrieve -> compose -> audit.
    Deterministic stages (triage, retrieve, audit) run locally in Python (0 LLM tokens);
    only compose invokes the LLM with compact context and top-k retrieved policies.
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        self._valid_articles = set(article_ids())

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        snapshot = json.dumps(order, separators=(",", ":"))

        triage = self._triage(thread, order)
        retrieved = self._retrieve(triage)
        draft = self._compose(ticket_id, thread, snapshot, retrieved)
        return self._audit(draft)

    def _triage(self, thread: str, order: dict) -> str:
        """Local triage: build a compact search query from thread + key order attributes."""
        order_signals = " ".join(f"{k} {v}" for k, v in order.items())
        return f"{thread} {order_signals}"

    def _retrieve(self, triage_query: str, max_k: int = 2) -> str:
        """Local retrieval: fetch only the top-k relevant KB articles (no 210k filler blob)."""
        return retrieve_top_articles(triage_query, max_k=max_k)

    def _compose(
        self,
        ticket_id: str,
        thread: str,
        snapshot: str,
        retrieved: str,
    ) -> str:
        """Single LLM call: decide action, kb_article, and customer_reply in JSON."""
        messages = [
            ChatMessage("system", _COMPOSE_SYSTEM),
            ChatMessage(
                "user",
                "\n\n".join(
                    [
                        f"TICKET_ID: {ticket_id}",
                        f"POLICIES:\n{retrieved}",
                        f"THREAD:\n{thread}",
                        f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
                    ]
                ),
            ),
        ]
        return self.llm.chat("compose", messages)

    def _audit(self, draft: str) -> TicketResolution:
        """Local guardrail: validate JSON schema, allowed action, and valid KB article ID."""
        parsed = parse_resolution(draft)
        if parsed.action not in _VALID_ACTIONS:
            raise ValueError(f"Audit failed: invalid action '{parsed.action}'")
        if parsed.kb_article not in self._valid_articles:
            raise ValueError(f"Audit failed: unknown kb_article '{parsed.kb_article}'")
        return TicketResolution(parsed.action, parsed.kb_article)