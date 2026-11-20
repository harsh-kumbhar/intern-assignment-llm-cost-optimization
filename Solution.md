# ShopAssist — LLM Cost Optimization Solution

## 1. Executive Summary (Before vs. After)

| Metric | Question 1 (`Single Call`) | Question 2 (`4-Node Pipeline`) |
| :--- | :---: | :---: |
| **Baseline Total (4 Public Tickets)** | 35,535 pts | 293,085 pts |
| **Optimized Total (4 Public Tickets)** | **4,953 pts** | **4,953 pts** |
| **Baseline Avg / Ticket** | ~8,883.8 pts | ~73,271.3 pts |
| **Optimized Avg / Ticket** | **1,238.2 pts** | **1,238.2 pts** |
| **Assignment Target / Ticket** | ≤ 3,000 pts | ≤ 2,500 pts |
| **Cost Reduction** | **-86.1%** | **-98.3%** |
| **Execution Time (4 Tickets)** | 0.51s → **0.034s** | 312.96s (5m 12s) → **0.011s** |
| **Accuracy (Public + Holdout `T-1091`)** | 5 / 5 Exact Match | 5 / 5 Exact Match |

---

## 2. Problems Identified in the Baseline

Under the billing model (`uncached_input × 3 + cached_input × 0.3 + output × 15`), profiling the starter code revealed five primary cost and latency bottlenecks:

1. **Output Token Bloat (`15 pts/token` — 5× input cost):** Both agents explicitly prompted the model with verbose triggers (`"step by step"`, `"full reasoning"`, `"restate the ticket"`, `"double-check"`). This forced ~480+ token completions (~7,200+ wasted points per call) even though downstream parsing only consumes a compact JSON object (`action`, `kb_article`, `customer_reply`).
2. **Cache-Busting Dynamic Prefixes:** Both `question1/agent.py` (`Run id {uuid.uuid4()}`) and `question2/agent.py` (`trace-{uuid.uuid4()}`) injected a random UUID on line 1 of the system prompt, guaranteeing 0% prefix-cache reuse and causing non-deterministic token counts across runs.
3. **Massive Unfiltered Context & 5-Minute Filler Lag (Q2):**
   * Q1 injected the entire Help Center (`all_articles_blob()`) into every call.
   * Q2 called `expanded_help_center_blob()` in `__init__()`, which builds a **210,000-token filler blob** in a loop while re-counting tokens on every iteration—causing a **5m 12s** startup bottleneck—and then dumped 120k characters into `triage` and 80k characters into `audit`.
4. **Redundant Multi-Call Pipeline & Context Snowballing (Q2):** Running 4 sequential LLM calls (`triage → retrieve → compose → audit`) multiplied the `15 pts/token` output penalty and re-passed duplicated context (e.g., `_compose` literally pasted `TRIAGE` twice).
5. **Serialization Overhead:** Pretty-printing order snapshots with `json.dumps(order, indent=2)` added unnecessary whitespace and newline tokens on every call.

---

## 3. Step-by-Step Methodology & Implementation

### Step 1: Baseline Capture
Before modifying any code, we ran the unmodified test suite and both `run.py` scripts with PowerShell's `Measure-Command` to record exact baseline points and latency:
* `pytest -q` → `6 passed in 333.61s (0:05:33)`
* `python question1/run.py` → `total_points: 35535` (`0.508s`)
* `python question2/run.py` → `total_points: 293085` (`312.960s` / `5m 12.96s`)

### Step 2: Question 1 Optimization (`question1/agent.py`)
* **Structured JSON-Only Prompting:** Removed the random UUID and all verbose reasoning hints. Replaced them with a concise system instruction requesting only valid JSON (`action`, `kb_article`, `customer_reply`), cutting output tokens from ~480+ down to **36–46 tokens**.
* **Confidence-Gated Adaptive Retrieval (`retrieve_top_articles`):** Replaced the full Help Center dump with a lightweight lexical retriever that scores KB articles against the ticket thread and order attributes (weighting article ID/title matches 3× higher than body matches).
  * If the #1 article clears a confidence margin (`score_1 >= score_2 + 3`), only the **top-1** policy is injected (`T-1002`, `T-1003`, `T-1004`).
  * If scores are close, it safely falls back to **top-2** (`T-1001`).
* **Compact Serialization:** Switched order snapshots to single-line JSON via `json.dumps(order, separators=(",", ":"))`.
* **Architectural Tradeoff — Why Adaptive Retrieval Over Prompt Caching?** Prompt caching requires an identical prefix of at least `1,024` tokens (`3,072` pts cold, `307+` pts warm). Confidence-gated retrieval keeps the entire prompt under **~255 tokens** (~600–750 input pts), which is cheaper on cold starts and low-volume runs while remaining 100% production-realistic.

### Step 3: Question 2 Optimization (`question2/agent.py`)
* **LLM-as-Composer, Code-as-Guardrail:** Preserved the clean 4-stage pipeline (`triage → retrieve → compose → audit`) while moving deterministic stages into local Python (`0` LLM tokens):
  1. `_triage(thread, order)`: Extracts structured search signals from the customer thread and order fields locally.
  2. `_retrieve(triage_query)`: Runs adaptive lexical retrieval directly over `shared/kb.py` (`article_ids()` / `load_article()`), completely bypassing the 210k-token filler blob and cutting runtime from **5m 12s to 0.011s**.
  3. `_compose(...)`: The **single LLM call**, receiving only the compact order snapshot, thread, and top-k retrieved policies.
  4. `_audit(draft)`: Deterministic guardrail that parses JSON and validates both the allowed `action` enum (`REFUND`, `REPLACE`, `SEND_ARTICLE`, `ESCALATE_TIER2`) and that `kb_article` exists in the Help Center.

---

## 4. Optimized Runs & Per-Call Token Breakdowns

### Question 1 — Per-Call Breakdown (Required: 1 Ticket + Full Public Summary)
**Detailed Breakdown (`T-1001`):**
| Ticket | Scenario | Call Label | Input Tokens | Cached Tokens | Output Tokens | Points | Target |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`T-1001`** | Refund, physical item, unopened | `handle_ticket` | 252 | 0 | 46 | **1,446** | ≤ 3,000 |

**All Public Tickets (Q1):**
| Ticket | Action | KB Article | Input | Cached | Output | Points |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `T-1001` | `REFUND` | `returns-within-30-days` | 252 | 0 | 46 | **1,446** |
| `T-1002` | `SEND_ARTICLE` | `digital-license-no-refund` | 207 | 0 | 39 | **1,206** |
| `T-1003` | `ESCALATE_TIER2` | `custom-items-no-replace` | 195 | 0 | 39 | **1,170** |
| `T-1004` | `REPLACE` | `damage-replacement-7-days` | 197 | 0 | 36 | **1,131** |
| **Total / Avg** | — | — | — | — | — | **4,953 (1,238.2 avg)** |

---

### Question 2 — Per-Call Breakdown (Required: 2 Tickets — Fulfillment & Denial/Escalation)
| Path Type | Ticket | Action | KB Article | Call Label | Input | Cached | Output | Points | Target |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Physical Fulfillment** | **`T-1004`** | `REPLACE` | `damage-replacement-7-days` | `compose` | 197 | 0 | 36 | **1,131** | ≤ 2,500 |
| **Denial / Escalation** | **`T-1003`** | `ESCALATE_TIER2` | `custom-items-no-replace` | `compose` | 195 | 0 | 39 | **1,170** | ≤ 2,500 |

*(Note: `triage`, `retrieve`, and `audit` consume **0 LLM tokens** and **0 points**. All 4 public tickets match Q1's token counts: `T-1001`: 1,446 | `T-1002`: 1,206 | `T-1003`: 1,170 | `T-1004`: 1,131; Total: **4,953 pts** in **0.011s**.)*

---

## 5. Test Suite & Verification Results

### A. Optimization, Budget & Guardrail Suite (`tests/test_optimization.py`)
Added 5 automated tests verifying exact graded accuracy (including holdout `T-1091`), per-ticket cost ceilings, single-call enforcement in Q2, retrieval relevance, and deterministic audit rejections:

```text
tests/test_optimization.py::test_q1_accuracy_and_budget PASSED                       [ 20%]
tests/test_optimization.py::test_q2_accuracy_and_budget PASSED                       [ 40%]
tests/test_optimization.py::test_adaptive_retrieval_returns_relevant_policy PASSED   [ 60%]
tests/test_optimization.py::test_q2_audit_rejects_invalid_action PASSED              [ 80%]
tests/test_optimization.py::test_q2_audit_rejects_unknown_kb_article PASSED          [100%]

5 passed in 0.39s


## 6. How to Reproduce & Evaluate

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Optimization, Budget & Guardrail Tests (~0.4s)
Verifies exact graded outputs (`action`, `kb_article`) on public tickets (`T-1001`–`T-1004`) plus holdout `T-1091`, per-ticket cost ceilings (`<= 3,000` for Q1, `<= 2,500` for Q2), single-call enforcement in Q2, retrieval relevance, and audit guardrails:
```bash
pytest -v tests/test_optimization.py
```

### 3. View Per-Call Token & Points Breakdowns (~0.05s)
Prints the formatted table of input/cached/output tokens, points, target status, and runtime vs. baseline:
```bash
python benchmark.py
```

### 4. Run the Official Assignment Entrypoints
Confirm the standard JSON output and total points directly:
```bash
python question1/run.py
python question2/run.py
```
*Expected `total_points` for both: **`4953`** (avg **1,238.2 pts/ticket**).*

### 5. Run the Original Starter Test Suite
All unmodified starter tests remain intact:
```bash
# Passes all 9 accuracy & optimization tests (skips the 2 unoptimized-baseline cost assertions)
pytest -q -k "not baseline_cost"
```
*(Running bare `pytest -q` produces **4 passed, 2 failed**, where the only 2 failures are `test_baseline_cost_is_high` asserting the unoptimized costs `> 15,000` and `> 70,000`.)*

### 6. Files Changed
* **Modified:** `question1/agent.py`, `question2/agent.py` (no imports from `eval/reference.py`).
* **Added:** `tests/test_optimization.py`, `benchmark.py`, `SOLUTION.md`.
* **Untouched:** `shared/*`, `eval/*`, `data/*`, `tests/test_question1.py`, `tests/test_question2.py`.