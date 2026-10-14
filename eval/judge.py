"""Judge the model's findings by hand: would you want this comment on your PR?

The test split's gold labels come from the teacher model and are noisy, so precision against them
undersells (or oversells) the model. This asks a human instead, one predicted finding at a time.

  python eval/evaluate.py --data test.jsonl --no-rules --backend llamacpp --model ... --dump preds.jsonl
  python eval/judge.py --data test.jsonl --preds preds.jsonl          # resumable, Ctrl-C any time

Keys:  y  useful, right category     c  useful, wrong category     n  not useful (noise / wrong)
       s  skip                        q  save & quit
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import hunk_from_user  # noqa: E402

KEYS = {"y": "useful", "c": "useful-wrong-category", "n": "not-useful"}


def report(judged: dict) -> None:
    total, by_cat = Counter(), {}
    for j in judged.values():
        total[j["verdict"]] += 1
        by_cat.setdefault(j["category"], Counter())[j["verdict"]] += 1
    n = sum(total.values())
    if not n:
        print("nothing judged yet")
        return
    useful = total["useful"] + total["useful-wrong-category"]
    print(f"\njudged {n} findings: useful {useful}/{n} = {useful / n:.0%}   "
          f"(right category {total['useful']}, wrong category {total['useful-wrong-category']}, "
          f"not useful {total['not-useful']})")
    print(f"{'category (as predicted)':<26}{'useful':>8}{'judged':>8}")
    for cat, c in sorted(by_cat.items()):
        u = c["useful"] + c["useful-wrong-category"]
        print(f"{cat:<26}{u / sum(c.values()):>8.0%}{sum(c.values()):>8}")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--data", required=True, help="the JSONL that was evaluated (same order as --preds)")
    ap.add_argument("--preds", required=True, help="evaluate.py --dump output")
    ap.add_argument("--out", default="data/tmp/judgments.jsonl")
    ap.add_argument("--show-gold", action="store_true", help="show the teacher's label after you answer")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    preds = [json.loads(l) for l in open(a.preds, encoding="utf-8")]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    judged = {}
    if out.exists():
        for l in out.open(encoding="utf-8"):
            j = json.loads(l)
            judged[j["key"]] = j

    todo = []
    for i, (row, p) in enumerate(zip(rows, preds)):
        for k, f in enumerate(p["pred"]):
            key = f"{i}:{k}:{f['line']}:{f['category']}"
            if key not in judged:
                todo.append((key, row, p, f))
    print(f"{len(todo)} findings to judge ({len(judged)} already done) → {out}")

    with out.open("a", encoding="utf-8") as fh:
        for n, (key, row, p, f) in enumerate(todo, 1):
            hunk = hunk_from_user(row["messages"][1]["content"])
            span = range(f["line"], f["end_line"] + 1)
            print("\n" + "=" * 100 + f"\n[{n}/{len(todo)}]  {hunk.path}")
            for ln, kind, text in hunk.lines:
                mark = ">>" if ln in span and kind == "+" else "  "
                print(f"{mark}{ln:>6} {kind} {text}")
            print(f"\n  MODEL: lines {f['line']}-{f['end_line']}  {f['category']} ({f['severity']})\n"
                  f"         {f['reason']}")
            while True:
                try:
                    ans = input("  useful? [y/c/n/s/q] ").strip().lower()
                except EOFError:
                    ans = "q"
                if ans in KEYS or ans in ("s", "q"):
                    break
            if ans == "q":
                break
            if ans == "s":
                continue
            j = {"key": key, "path": hunk.path, "category": f["category"], "verdict": KEYS[ans]}
            judged[key] = j
            fh.write(json.dumps(j) + "\n")
            fh.flush()
            if a.show_gold:
                print(f"  (teacher said: {[(g['line'], g['category'], g['reason']) for g in p['gold']] or 'clean'})")
    report(judged)


if __name__ == "__main__":
    main()
