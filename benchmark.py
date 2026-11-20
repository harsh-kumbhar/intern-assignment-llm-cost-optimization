\import time

from question1.agent import SupportCopilot
from question2.agent import SupportPipeline

PUBLIC = ["T-1001", "T-1002", "T-1003", "T-1004"]


def run_report(name: str, agent_cls, target_per_ticket: int, baseline_total: int, baseline_sec: float):
    print(f"\n{'=' * 78}")
    print(f" {name} (Target: <= {target_per_ticket} pts/ticket)")
    print(f"{'=' * 78}")
    print(f"{'Ticket':<8} {'Action':<15} {'KB Article':<27} {'In':>4} {'Cache':>5} {'Out':>4} {'Points':>7} {'Status':>7}")
    print("-" * 78)

    t0 = time.perf_counter()
    total_pts = 0

    for tid in PUBLIC:
        agent = agent_cls()
        res = agent.handle_ticket(tid)
        pts = agent.session.total_points()
        total_pts += pts
        c = agent.session.calls[0]
        status = "PASS" if pts <= target_per_ticket else "FAIL"
        print(
            f"{tid:<8} {res.action:<15} {res.kb_article:<27} "
            f"{c.input_tokens:>4} {c.cached_prefix_tokens:>5} {c.output_tokens:>4} "
            f"{pts:>7} {status:>7}"
        )

    elapsed = time.perf_counter() - t0
    avg_pts = total_pts / len(PUBLIC)
    reduction = (1 - total_pts / baseline_total) * 100

    print("-" * 78)
    print(f"Total Points : {total_pts:,} pts (Baseline: {baseline_total:,} pts | -{reduction:.1f}%)")
    print(f"Avg / Ticket : {avg_pts:,.1f} pts (Target: <= {target_per_ticket:,} pts)")
    print(f"Elapsed Time : {elapsed:.3f}s (Baseline: {baseline_sec:.2f}s)")


if __name__ == "__main__":
    run_report("Question 1 — Single Call", SupportCopilot, target_per_ticket=3000, baseline_total=35535, baseline_sec=0.51)
    run_report("Question 2 — 4-Node Pipeline", SupportPipeline, target_per_ticket=2500, baseline_total=293085, baseline_sec=312.96)