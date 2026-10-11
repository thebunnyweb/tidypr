# tidypr: plan v1.0

A small fine-tuned model that reviews pull request diffs for "AI slop" (bloat, needless defensiveness, filler) and posts focused, low-noise comments from a GitHub Action. It runs entirely on the CI runner, costs nothing per PR, and code never leaves CI.

> Working name: `tidypr` (rename freely). Lives in its own new repo, separate from `scrubr`.

---

## 1. Goal and non-goals

**Goal:** catch quality problems that AI coding agents commonly add and that linters miss, with very few false positives.

**Non-goals**
- Detecting *whether AI wrote the code*. We flag code problems, not authors.
- Finding deep logic bugs or replacing a human or large-LLM reviewer.
- Anything ruff/pylint already covers (unused imports, bare `except:`, formatting). We defer to them.

## 2. Locked decisions

| Area | Decision |
|---|---|
| Repo | New standalone repo |
| Language scope v1 | **Python only** (others in v2) |
| Base model | **Qwen2.5-Coder-1.5B-Instruct** (Apache-2.0). Fallbacks: Qwen2.5-Coder-7B, Qwen3-4B. Llama 3.2-3B kept only as a baseline. *Check for a newer small coder model at kickoff.* |
| Training | **Unsloth**, QLoRA, on a **free Kaggle GPU** (T4/P100). The Mac is used for development only; optional `mlx-lm` smoke tests. |
| Inference | GGUF Q4_K_M (~1 GB) under **llama.cpp** (`llama-cpp-python`) on the GitHub runner's CPU. No Ollama. |
| Output | Structured JSON findings with a suggested fix (not just a flag) |
| Budget | **$0–15 expected**, hard cap $50 |

## 3. Slop taxonomy v1

| Category | Flag when | Default severity |
|---|---|---|
| `over-defensive` | Catch-all `except Exception` that swallows or hides errors; `None`/type checks on values that can't be `None`; redundant if/else around trivial returns | medium–high |
| `redundant-comment` | A comment restates the next line(s) of code | low |
| `verbose-docstring` | Docstring far longer than the function, filler phrases ("robust", "comprehensive", "This function is responsible for") | low |
| `needless-abstraction` | A class or factory wrapping one function, unused parameters or config, a single-use helper that only renames | medium |
| `dead-or-placeholder` | `# TODO: implement`, `pass` stubs, commented-out code blocks (things ruff can't see) | medium |
| `log-noise` | Emoji or celebratory logs, `print` debugging in library or service code (not CLIs) | low |
| `hallucinated-api` | A call to a method or argument that doesn't exist for a known library | high |
| `pr-description-fluff` | PR body full of adjectives with no concrete change list | low (summary only) |

Each finding:
```json
{"line": 26, "end_line": 27, "category": "over-defensive", "severity": "high",
 "reason": "except Exception: pass hides send_report failures.",
 "suggestion": "except SMTPException:\n    logger.exception(\"report failed\")"}
```
An empty list `[]` is a valid and common answer.

## 4. Data

### Sources (all free)
| Source | Role | Notes |
|---|---|---|
| **AIDev** (`hao-li/AIDev`, HF) | Real AI-agent PRs and, in the curated subset of about 33k, reviewer comments | Main source of real slop; check the licence and terms at download |
| **CommitPackFT** (bigcode) | Pre-2023 human commits under permissive licences | Clean negatives, and the base for synthetic slop |
| **CodeReviewer** (Microsoft) | Hunk-to-review-comment pairs | Teaches comment style and format; check the licence |
| Your own repos | Real PRs | Demo plus extra gold examples |

### Labelling strategy
1. **Rule labels:** AST/regex checks auto-label the easy categories (catch-all excepts, TODO stubs, emoji logs, comment-equals-code similarity).
2. **Reviewer-comment labels:** AIDev review comments like "unnecessary", "remove", "too verbose" anchored to a line become weak labels, mapped to categories by keyword plus a small model.
3. **Synthetic injection:** take a clean CommitPackFT hunk and have an LLM inject 1–3 *named* categories. The label is known, and the original hunk is the fix. Use a free tier or a local model first, and Claude Batch only if quality is poor (capped at $10–15).
4. **Gold set (hand-labelled, about 300 hunks):** a mix of real AIDev hunks and clean human hunks. **Never trained on.** About 4–6 hours of labelling using a small labelling tool we'll build.

### Target mix
About 10k training examples: ~45% clean (`[]`), ~35% synthetic slop, ~20% real slop with weak labels. Held-out validation is 5% of the training sources; the test set is the gold set only.

Split by **repository** (never by hunk) so the model can't memorise repos.

## 5. Training (Unsloth on Kaggle)

Starting configuration, to be tuned:
- `load_in_4bit=True`, LoRA `r=16`, `alpha=32`, dropout 0, targets set to all linear layers
- `max_seq_length=4096`, about 1 hunk plus ~20 lines of file context per example
- lr `2e-4`, cosine, 2–3 epochs, effective batch 16, `train_on_responses_only`
- Chat template: system (taxonomy plus rules), then user (file path, language, hunk with line numbers), then assistant (JSON)
- Export: merged → GGUF `Q4_K_M` (plus `Q8_0` for comparison)

Expected cost: about 1–2 hours per run on a free T4. Plan for 3–5 runs.

## 6. Evaluation and go/no-go gates

Measured on the gold set only:

| Metric | Gate for v0.1 release |
|---|---|
| Precision on medium and high findings | **≥ 0.80** |
| False-positive rate on clean human hunks | **≤ 5%** |
| Valid-JSON rate | ≥ 99% |
| Macro-F1 versus the rules-only baseline | +10 points or more |
| Macro-F1 versus zero-shot Qwen-1.5B | +15 points or more |
| Median PR runtime on a standard runner | ≤ 3 min |

Reference ceiling: a large model (e.g. Claude) zero-shot on the same gold set, to see how close we get. That costs about $5.

If the gates fail: (1) improve the data, then (2) try 7B, then (3) narrow the categories. In that order.

## 7. GitHub Action

Pipeline: get the PR diff, run the rules pass, run the model on the changed Python hunks (cached GGUF), merge and deduplicate, filter by severity and cap, then post.

Posting:
- **One review** with inline comments using ` ```suggestion ` blocks (only medium and above by default)
- **One sticky summary comment**, edited in place via a `<!-- tidypr:summary -->` marker
- **One check run** `tidypr`, which fails on `fail_on` severity
- A fingerprint per finding to avoid duplicates; threads auto-resolve when a finding disappears
- PRs from forks: fall back to `::warning` annotations (read-only token)

Inputs: `min_severity`, `fail_on`, `max_findings`, `categories`, `post_mode` (`summary-only` | `review` | `annotations`).

Feedback loop: 👍/👎 reactions on bot comments are exported and become the next round's training data.

## 8. Repo layout

```
tidypr/
├── rules/            # Phase 0 AST/regex detectors (zero-dep)
├── data/
│   ├── fetch/        # AIDev, CommitPackFT, CodeReviewer downloaders + filters
│   ├── label/        # rule labels, reviewer-comment mapping, synthetic injection
│   ├── gold/         # hand-labelled test set + labelling tool
│   └── build.py      # → train.jsonl / val.jsonl (repo-split)
├── training/
│   ├── kaggle_unsloth.ipynb
│   └── export_gguf.py
├── eval/             # metrics, baselines, gate report
├── action/           # action.yml, entrypoint, GitHub posting, dedup
└── docs/PLAN.md
```

## 9. Milestones

| # | Milestone | Done when |
|---|---|---|
| M0 | Scaffold + rules pass | Rules run over a local diff and output JSON findings; tests pass |
| M1 | Data pipeline | `train.jsonl`/`val.jsonl` built from all sources; the dataset card records counts per category |
| M2 | Gold set + baselines | About 300 hand-labelled hunks; rules, zero-shot and Claude-ceiling scores recorded |
| M3 | Fine-tune v0.1 | Kaggle run, GGUF exported, gate report produced |
| M4 | Action in `summary-only` mode | Running on your own repos (incl. scrubr) for one week |
| M5 | `review` mode + feedback loop | Inline suggestions live; reaction export feeding v0.2 data |

## 10. Budget

| Item | Expected | Cap |
|---|---|---|
| Kaggle training | $0 | — |
| Synthetic data (free tier / local first) | $0–10 | $15 |
| Claude ceiling eval | ~$5 | $5 |
| Optional RunPod (7B fallback) | $0 | $15 |
| **Total** | **$0–15** | **$50** |

## 11. Risks

| Risk | Mitigation |
|---|---|
| Synthetic slop doesn't match real slop | Gold set is 100% real; increase the share of real weak labels if the gap is large |
| Too noisy, so developers ignore it | Precision gate, severity thresholds, `summary-only` first week, hard cap |
| `hallucinated-api` needs knowledge a 1.5B model lacks | Treat as best effort; may move to a rule that checks against installed packages in v2 |
| Kaggle quotas or session limits | Checkpoint every N steps; runs are short anyway |
| Dataset licence or terms | Check AIDev and CodeReviewer terms at M1; CommitPackFT is permissive-only |
| Long PRs are slow on CPU | Changed Python hunks only, a per-PR hunk cap, cached model |
