"""Re-apply the current teacher validation to stored teacher outputs (no GPU needed).

  python data/real/relabel.py --labeled data/real/labeled.jsonl --out data/real/labeled_v2.jsonl

Rows the teacher already accepted keep their finding (re-checked); rejected rows are re-parsed from
their raw output with the newer, repairing parser. Prints where examples are lost and why.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from teacher import _EXPLICIT, consistent, parse_teacher_explain  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    before, after, why, cats = Counter(), Counter(), Counter(), Counter()
    with open(a.out, "w", encoding="utf-8") as fh:
        for p in a.labeled:
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                before[r.get("label")] += 1
                label, finding, w = parse_teacher_explain(r.get("raw", ""), r)
                if r.get("label") == "slop" and r.get("finding") and label != "slop":
                    # raw may have been truncated in older runs: re-check the stored finding instead
                    f = r["finding"]
                    texts = [t for n, k, t in r["hunk"] if f["line"] <= n <= f["end_line"]]
                    vague = f["category"] in ("needless-abstraction", "hallucinated-api") \
                        and not _EXPLICIT.search(r.get("comment") or "")
                    if not vague and consistent(f["category"], f["reason"], texts):
                        label, finding, w = "slop", f, ""
                    elif vague:
                        label, finding, w = "not_slop", None, "vague-comment"
                r.update(label=label, finding=finding, why=w)
                after[label] += 1
                if w:
                    why[w] += 1
                if finding:
                    cats[finding["category"]] += 1
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("before:", dict(before))
    print("after :", dict(after))
    print("why rejected/changed:", dict(why.most_common()))
    print("slop categories:", dict(cats.most_common()))


if __name__ == "__main__":
    main()
