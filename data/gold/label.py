"""Hand-label real PR hunks into a gold test set (same JSONL format as training data).

  git -C ~/code/some-repo log -p --since=2025-01-01 -- '*.py' > /tmp/real.diff
  python data/gold/label.py /tmp/real.diff                 # appends to data/gold/gold.jsonl

For each hunk:
  Enter                 → clean (no findings)        s → skip     q → save & quit
  or one finding per line, then an empty line:
      <line>[-<end>] <cat> <sev> <reason>
      e.g.  27-28 od h swallows send_report errors
  cat: od over-defensive · rc redundant-comment · vd verbose-docstring · na needless-abstraction
       dp dead-or-placeholder · ln log-noise · ha hallucinated-api
  sev: l / m / h

Never train on this file. It's the honest test: python eval/evaluate.py --data data/gold/gold.jsonl ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tidypr.diff import code_hunks, parse_diff  # noqa: E402
from tidypr.prompt import build_messages  # noqa: E402
from tidypr.rules import check_hunk  # noqa: E402
from tidypr.schema import Finding  # noqa: E402

CATS = {"od": "over-defensive", "rc": "redundant-comment", "vd": "verbose-docstring", "na": "needless-abstraction",
        "dp": "dead-or-placeholder", "ln": "log-noise", "ha": "hallucinated-api"}
SEVS = {"l": "low", "m": "medium", "h": "high"}


def parse_finding(line: str, path: str, valid: set[int]) -> Finding:
    span, cat, sev, reason = line.split(maxsplit=3)
    a, _, b = span.partition("-")
    start, end = int(a), int(b or a)
    if cat not in CATS or sev not in SEVS:
        raise ValueError(f"category must be one of {list(CATS)} and severity one of {list(SEVS)}")
    if not {start, end} <= valid:
        raise ValueError("lines must be added (+) lines of this hunk")
    return Finding(path, start, CATS[cat], SEVS[sev], reason.strip(), end_line=end, source="gold")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("diff")
    ap.add_argument("--out", default=str(Path(__file__).parent / "gold.jsonl"))
    ap.add_argument("--max-lines", type=int, default=60, help="split longer hunks")
    ap.add_argument("--include-tests", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    seen = set()
    if out.exists():
        seen = {json.loads(l).get("id") for l in out.open(encoding="utf-8")}
    hunks = [s for h in code_hunks(parse_diff(Path(a.diff).read_text(errors="replace")),
                                     skip_tests=not a.include_tests)
             for s in h.split(a.max_lines)]
    todo = []
    for h in hunks:
        hid = hashlib.sha1((h.path + h.render()).encode()).hexdigest()[:16]
        if hid not in seen:
            todo.append((hid, h))
    print(f"{len(hunks)} code hunks, {len(todo)} not labelled yet → {out}\n")

    done = 0
    with out.open("a", encoding="utf-8") as fh:
        for i, (hid, h) in enumerate(todo, 1):
            print("=" * 100 + f"\n[{i}/{len(todo)}]  File: {h.path}\n" + h.render())
            hints = check_hunk(h)
            if hints:
                print("\n  rule hints: " + "; ".join(f"{f.line} {f.category}" for f in hints))
            findings: list[Finding] = []
            while True:
                try:
                    raw = input("> ").strip()
                except EOFError:
                    raw = "q"
                if raw == "q":
                    print(f"\nsaved {done} labels this session.")
                    return
                if raw == "s" and not findings:
                    break
                if not raw:
                    fh.write(json.dumps({"id": hid, "messages": build_messages(h, findings)}, ensure_ascii=False) + "\n")
                    fh.flush()
                    done += 1
                    print(f"  ✓ saved ({'clean' if not findings else f'{len(findings)} finding(s)'})")
                    break
                try:
                    findings.append(parse_finding(raw, h.path, h.added_lines))
                    print(f"  + {findings[-1].category} {findings[-1].severity} at {findings[-1].line}"
                          " (more findings, or Enter to save)")
                except ValueError as e:
                    print(f"  ! {e}")
    print(f"\nall done: {done} labelled.")


if __name__ == "__main__":
    main()
