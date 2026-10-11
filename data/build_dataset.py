"""Build train/valid JSONL in chat format ({"messages": [...]}): works for both Unsloth and MLX.

Sources (combine freely; files are interleaved round-robin so no single source dominates):
  --source DIR[=MAX]        any folder of human-written .py files, optionally capped at MAX examples.
                            Your own pre-2023 repos are the best match for your PRs.
  --commitpackft N[=MAX]    stream N Python files from bigcode/commitpackft (needs `pip install datasets`)

Mix (v2): clean (untouched code) / decoy (looks like slop, isn't) / slop (injected, labelled).

Examples:
  STDLIB="$(python -c 'import sysconfig;print(sysconfig.get_paths()["stdlib"])')"
  python data/build_dataset.py --out data/v2 --max-examples 6000 \\
      --commitpackft 40000=3500 --source ~/code/old-rasa-bot=1000 --source "$STDLIB=1500"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tidypr.prompt import build_messages  # noqa: E402
from tidypr.synth import extract_functions, make_example  # noqa: E402

SKIP_DIRS = {"site-packages", "__pycache__", "idlelib", ".venv", "venv", "node_modules", "build", "dist", ".git",
             ".tox", "migrations"}


def iter_source_dir(root: str, seed: int = 0):
    root_p = Path(root).expanduser()
    paths = sorted(root_p.rglob("*.py"))
    random.Random(seed).shuffle(paths)  # don't let alphabetical order decide what fits under the cap
    for p in paths:
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        try:
            # group = top-level folder, so a package never straddles train/valid
            rel = p.relative_to(root_p)
            yield f"{root_p.name}/{rel.parts[0]}::{rel}", p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


def iter_commitpackft(n: int):
    from datasets import load_dataset

    ds = load_dataset("bigcode/commitpackft", "python", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= n:
            break
        # group key = repo, so the split never leaks a repo across train/valid
        yield f"{row['repos'].split(',')[0]}::{row['new_file']}", row["new_contents"]


def split_of(group: str, valid_frac: float) -> str:
    h = int(hashlib.md5(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "valid" if h < valid_frac else "train"


def _spec(s: str) -> tuple[str, int | None]:
    if "=" in s and s.rsplit("=", 1)[1].isdigit():
        v, cap = s.rsplit("=", 1)
        return v, int(cap)
    return s, None


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--source", "--source-dir", dest="source", action="append", default=[])
    ap.add_argument("--commitpackft", default=None, help="N or N=MAX")
    ap.add_argument("--out", default="data/out")
    ap.add_argument("--mix", default="0.30,0.20,0.50", help="clean,decoy,slop fractions")
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--max-examples", type=int, default=6000)
    ap.add_argument("--per-file", type=int, default=8, help="max functions per file, for diversity")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    weights = [float(x) for x in a.mix.split(",")]
    sources = []
    for s in a.source:
        d, cap = _spec(s)
        sources.append({"name": d, "it": iter_source_dir(d, a.seed), "cap": cap, "n": 0})
    if a.commitpackft:
        n, cap = _spec(a.commitpackft)
        sources.append({"name": "commitpackft", "it": iter_commitpackft(int(n)), "cap": cap, "n": 0})
    if not sources:
        ap.error("give at least one --source DIR or --commitpackft N")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "valid")}
    stats: Counter = Counter()

    active = list(sources)
    while active and stats["total"] < a.max_examples:
        for src in list(active):  # round-robin: one file per source per pass
            if src["cap"] is not None and src["n"] >= src["cap"]:
                active.remove(src)
                continue
            try:
                key, code = next(src["it"])
            except StopIteration:
                active.remove(src)
                continue
            group, path = key.split("::", 1)
            funcs = extract_functions(code, path)
            rng.shuffle(funcs)
            for func in funcs[: a.per_file]:
                if stats["total"] >= a.max_examples or (src["cap"] is not None and src["n"] >= src["cap"]):
                    break
                kind = rng.choices(["clean", "decoy", "slop"], weights=weights)[0]
                ex = make_example(func, rng, kind)
                if ex is None:
                    continue
                hunk, findings = ex
                split = split_of(group, a.valid_frac)
                files[split].write(json.dumps({"messages": build_messages(hunk, findings)}, ensure_ascii=False) + "\n")
                src["n"] += 1
                stats["total"] += 1
                stats[split] += 1
                stats[f"kind:{kind}"] += 1
                stats[f"source:{Path(src['name']).name or src['name']}"] += 1
                stats["no-findings" if not findings else "with-findings"] += 1
                for f in findings:
                    stats[f"cat:{f.category}"] += 1

    for f in files.values():
        f.close()
    (out / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
