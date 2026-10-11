"""Build train/valid JSONL in chat format ({"messages": [...]}) — works for both Unsloth and MLX.

Sources (combine freely):
  --source-dir DIR       any folder of human-written .py files (e.g. CPython's stdlib, offline, free)
  --commitpackft N       stream N Python commits from bigcode/commitpackft (needs `pip install datasets`)

Examples:
  python data/build_dataset.py --source-dir "$(python -c 'import sysconfig;print(sysconfig.get_paths()["stdlib"])')" \
      --out data/out --max-examples 4000
  python data/build_dataset.py --commitpackft 20000 --out data/out
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
from tidypr.synth import clean_example, extract_functions, slop_example  # noqa: E402


def iter_source_dir(root: str, seed: int = 0):
    paths = sorted(Path(root).rglob("*.py"))
    random.Random(seed).shuffle(paths)  # don't let alphabetical order decide what fits under --max-examples
    for p in paths:
        if any(part in {"test", "tests", "site-packages", "__pycache__", "idlelib"} for part in p.parts):
            continue
        try:
            yield str(p.relative_to(root)), p.read_text(encoding="utf-8")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", action="append", default=[])
    ap.add_argument("--commitpackft", type=int, default=0)
    ap.add_argument("--out", default="data/out")
    ap.add_argument("--clean-frac", type=float, default=0.45, help="share of examples with no findings")
    ap.add_argument("--valid-frac", type=float, default=0.05)
    ap.add_argument("--max-examples", type=int, default=10000)
    ap.add_argument("--per-file", type=int, default=15, help="max functions per file, for diversity")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    sources = [iter_source_dir(d, a.seed) for d in a.source_dir]
    if a.commitpackft:
        sources.append(iter_commitpackft(a.commitpackft))
    if not sources:
        ap.error("give at least one --source-dir or --commitpackft N")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "valid")}
    stats: Counter = Counter()

    for src in sources:
        for key, code in src:
            group = key.split("::")[0] if "::" in key else key
            path = key.split("::")[-1]
            if stats["total"] >= a.max_examples:
                break
            funcs = extract_functions(code, path)
            rng.shuffle(funcs)
            for func in funcs[: a.per_file]:
                if stats["total"] >= a.max_examples:
                    break
                ex = clean_example(func) if rng.random() < a.clean_frac else slop_example(func, rng)
                if ex is None:
                    continue
                hunk, findings = ex
                split = split_of(group, a.valid_frac)
                files[split].write(json.dumps({"messages": build_messages(hunk, findings)}, ensure_ascii=False) + "\n")
                stats["total"] += 1
                stats[split] += 1
                stats["clean" if not findings else "slop"] += 1
                for f in findings:
                    stats[f"cat:{f.category}"] += 1

    for f in files.values():
        f.close()
    (out / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
