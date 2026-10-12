"""Score a backend against labelled examples (chat-format JSONL, same as training data).

  python eval/evaluate.py --data data/out/valid.jsonl                          # rules baseline
  python eval/evaluate.py --data data/gold/gold.jsonl --backend llamacpp --model bunnyweb/tidypr-1.5b-gguf

A predicted finding matches a gold finding when the category is equal and the line spans overlap.
Gates from docs/PLAN.md §6 are printed at the end.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tidypr.diff import Hunk  # noqa: E402
from tidypr.infer import ReviewConfig, make_backend, review_hunks  # noqa: E402
from tidypr.schema import CATEGORIES, parse_model_output, severity_at_least  # noqa: E402

_ROW = re.compile(r"^\s*(\d+) ([+ ]) (.*)$")


def hunk_from_user(content: str) -> Hunk:
    path = content.split("\n", 1)[0].removeprefix("File: ").strip()
    body = content.split("```diff\n", 1)[1].rsplit("\n```", 1)[0]
    lines = [(int(m[1]), m[2], m[3]) for m in map(_ROW.match, body.split("\n")) if m]
    return Hunk(path, lines)


def overlap(a, b) -> bool:
    return a.category == b.category and spans_overlap(a, b)


def spans_overlap(a, b) -> bool:
    return a.span[0] <= b.span[1] and b.span[0] <= a.span[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--backend", default="rules", choices=["rules", "llamacpp", "mlx"])
    ap.add_argument("--model")
    ap.add_argument("--model-file", default="*Q8_0.gguf")
    ap.add_argument("--adapter")
    ap.add_argument("--no-rules", action="store_true", help="score the model alone")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", help="write gold vs predicted findings per example to this JSONL file")
    a = ap.parse_args()

    backend = make_backend(a.backend, a.model, filename=a.model_file, adapter_path=a.adapter)
    cfg = ReviewConfig(min_severity="low", max_findings=50, use_rules=not a.no_rules,
                       categories=frozenset(CATEGORIES))  # score every category, gating is a product choice
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    clean_total = clean_flagged = n = 0
    loc_tp = loc_pred = loc_gold_hit = loc_gold = 0   # "right lines, any category"
    dump = open(a.dump, "w", encoding="utf-8") if a.dump else None

    for line in open(a.data, encoding="utf-8"):
        if a.limit and n >= a.limit:
            break
        msgs = json.loads(line)["messages"]
        hunk = hunk_from_user(msgs[1]["content"])
        gold = parse_model_output(msgs[2]["content"], hunk.path)
        pred = review_hunks([hunk], backend, cfg)
        n += 1
        if not gold:
            clean_total += 1
            clean_flagged += any(severity_at_least(p.severity, "medium") for p in pred)
        matched = set()
        for p in pred:
            hit = next((i for i, g in enumerate(gold) if i not in matched and overlap(p, g)), None)
            if hit is None:
                fp[p.category] += 1
            else:
                matched.add(hit)
                tp[p.category] += 1
        for i, g in enumerate(gold):
            if i not in matched:
                fn[g.category] += 1
        loc_pred += len(pred)
        loc_tp += sum(any(spans_overlap(p, g) for g in gold) for p in pred)
        loc_gold += len(gold)
        loc_gold_hit += sum(any(spans_overlap(p, g) for p in pred) for g in gold)
        if dump:
            row = lambda f: {"line": f.line, "end_line": f.span[1], "category": f.category,
                             "severity": f.severity, "reason": f.reason}
            dump.write(json.dumps({"path": hunk.path, "gold": [row(g) for g in gold],
                                   "pred": [row(p) for p in pred]}, ensure_ascii=False) + "\n")

    print(f"{'category':<22}{'P':>7}{'R':>7}{'F1':>7}   tp/fp/fn")
    f1s = []
    for c in CATEGORIES:
        t, f, m = tp[c], fp[c], fn[c]
        if t + f + m == 0:
            continue
        p = t / (t + f) if t + f else 0.0
        r = t / (t + m) if t + m else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f1)
        print(f"{c:<22}{p:7.2f}{r:7.2f}{f1:7.2f}   {t}/{f}/{m}")
    T, F = sum(tp.values()), sum(fp.values())
    print(f"\nexamples: {n}   overall precision: {T / (T + F) if T + F else 0:.2f}   "
          f"macro-F1: {sum(f1s) / len(f1s) if f1s else 0:.2f}")
    print(f"location only (right lines, any category): precision {loc_tp / loc_pred if loc_pred else 0:.2f}   "
          f"recall {loc_gold_hit / loc_gold if loc_gold else 0:.2f}   ({loc_tp}/{loc_pred} predicted, "
          f"{loc_gold_hit}/{loc_gold} gold)")
    if dump:
        dump.close()
        print(f"predictions written to {a.dump}")
    fpr = clean_flagged / clean_total if clean_total else 0
    print(f"clean hunks flagged medium+: {clean_flagged}/{clean_total} = {fpr:.1%}  (gate ≤ 5%)")


if __name__ == "__main__":
    main()
