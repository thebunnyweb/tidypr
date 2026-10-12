"""Merge everything into the master dataset: train / valid / test (chat-format JSONL).

  python data/real/merge.py --out data/master \
      --labeled data/real/labeled.jsonl --clean data/real/clean.jsonl \
      --synthetic 3000 --source "$STDLIB=3000"

* real slop    teacher-labelled human review comments (AIDev, CodeReviewer), several languages
* real clean   hunks reviewers accepted without comment (CodeReviewer y=0)
* synthetic    v2 generator (clean / decoy / slop), Python only, train+valid only

The TEST split is real data only, held out by repo, so its score says how the model does on real PRs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "data" / "real"))

from build_dataset import _spec, iter_commitpackft, iter_source_dir  # noqa: E402
from tidypr.diff import Hunk  # noqa: E402
from tidypr.prompt import build_messages  # noqa: E402
from tidypr.rules import check_hunk  # noqa: E402
from tidypr.schema import Finding  # noqa: E402
from tidypr.synth import extract_functions, make_example  # noqa: E402


def bucket(key: str) -> float:
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


GATED = {"needless-abstraction", "hallucinated-api"}   # off by default at review time: keep them a minority


def load_real_slop(paths: list[str], max_cat_share: float = 0.35, seed: int = 13,
                   gated_share: float = 0.10) -> list[dict]:
    """One example per distinct hunk. Findings from *sibling* comments (same PR + file) are attached to
    every hunk that contains those exact lines, so a flagged line is never left unlabelled elsewhere."""
    from teacher import consistent, reconcile

    rows, recat = [], Counter()
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("label") == "slop" and r.get("finding"):
                f = r["finding"]
                cat = reconcile(f["category"], f["reason"])   # older labels predate this check
                if cat is None:
                    recat["dropped"] += 1
                    continue
                if cat != f["category"]:
                    recat[f"{f['category']} -> {cat}"] += 1
                    f = r["finding"] = {**f, "category": cat}
                texts = [t for n, k, t in r["hunk"] if f["line"] <= n <= f["end_line"]]
                if consistent(f["category"], f["reason"], texts):   # re-check older labels too
                    rows.append(r)

    if recat:
        print(f"real slop: category fixes {dict(recat.most_common())}", file=sys.stderr)

    def finding_of(r):
        f = r["finding"]
        return Finding(r["path"], f["line"], f["category"], f["severity"], f["reason"],
                       end_line=f["end_line"], fix=f.get("fix"), source="teacher")

    by_file: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by_file[(r["source"], r.get("pr") or r["id"], r["path"])].append(r)

    examples: dict[tuple, dict] = {}
    for sib in by_file.values():
        for r in sib:
            key = (r["source"], r["repo"], r["path"], json.dumps(r["hunk"]))
            if key in examples:
                continue
            hunk = Hunk(r["path"], [tuple(x) for x in r["hunk"]])
            plus = {n: t for n, k, t in hunk.lines if k == "+"}
            findings: list[Finding] = [finding_of(r)]   # this row's own finding, always
            for o in sib:   # + siblings whose exact lines are in this hunk
                f, olines = o["finding"], {n: t for n, k, t in o["hunk"]}
                span = range(f["line"], f["end_line"] + 1)
                if all(n in plus and plus[n] == olines.get(n) for n in span):
                    new = finding_of(o)
                    if not any(x.category == new.category and x.line <= new.span[1] and new.line <= x.span[1]
                               for x in findings):
                        findings.append(new)
            findings.sort(key=lambda f: f.line)
            examples[key] = {"repo": f"{r['source']}:{r['repo']}", "source": r["source"], "hunk": hunk,
                             "findings": findings, "lang": r.get("lang")}

    # cap any single category's share so the model doesn't learn "every comment is slop"
    out = [e for e in examples.values() if e["findings"]]
    print(f"real slop: {len(rows)} labelled rows -> {len(out)} distinct hunks", file=sys.stderr)
    random.Random(seed).shuffle(out)
    cap = max(1, int(max_cat_share * len(out)))
    gcap = max(1, int(gated_share * len(out)))
    kept, per_cat = [], Counter()
    for ex in out:
        main = ex["findings"][0].category
        limit = gcap if main in GATED else cap
        if per_cat[main] >= limit and len({f.category for f in ex["findings"]}) == 1:
            continue
        per_cat[main] += 1
        kept.append(ex)
    print(f"real slop: {len(kept)} after caps ({max_cat_share:.0%}, gated {gated_share:.0%}) {dict(per_cat)}",
          file=sys.stderr)
    return kept


def load_real_clean(paths: list[str], cap: int) -> list[dict]:
    out = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            h = Hunk(r["path"], [tuple(x) for x in r["hunk"]])
            if r["path"].endswith(".py") and check_hunk(h):
                continue  # "accepted" hunk that our own rules would flag: not a trustworthy clean example
            out.append({"repo": f"{r['source']}:{r['repo']}", "source": r["source"], "hunk": h, "findings": [],
                        "lang": r.get("lang")})
            if len(out) >= cap:
                return out
    return out


def synthetic(sources: list[str], commitpackft: str | None, n: int, mix: list[float], seed: int) -> list[dict]:
    rng = random.Random(seed)
    its = []
    for s in sources:
        d, cap = _spec(s)
        its.append([iter_source_dir(d, seed), cap or n, 0])
    if commitpackft:
        num, cap = _spec(commitpackft)
        its.append([iter_commitpackft(int(num)), cap or n, 0])
    out = []
    while its and len(out) < n:
        for it in list(its):
            try:
                key, code = next(it[0])
            except StopIteration:
                its.remove(it)
                continue
            group, path = key.split("::", 1)
            funcs = extract_functions(code, path)
            rng.shuffle(funcs)
            for func in funcs[:6]:
                if len(out) >= n or it[2] >= it[1]:
                    break
                kind = rng.choices(["clean", "decoy", "slop"], weights=mix)[0]
                ex = make_example(func, rng, kind)
                if ex:
                    out.append({"repo": f"synth:{group}", "source": f"synth-{kind}", "hunk": ex[0],
                                "findings": ex[1], "lang": "Python"})
                    it[2] += 1
            if it[2] >= it[1]:
                its.remove(it)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--out", default="data/master")
    ap.add_argument("--labeled", action="append", default=[])
    ap.add_argument("--clean", action="append", default=[])
    ap.add_argument("--clean-ratio", type=float, default=1.0, help="real clean examples per real slop example")
    ap.add_argument("--max-cat-share", type=float, default=0.35, help="cap any one category's share of real slop")
    ap.add_argument("--gated-share", type=float, default=0.10,
                    help="cap for categories that are off by default (needless-abstraction, hallucinated-api)")
    ap.add_argument("--synthetic", type=int, default=3000, help="max synthetic examples")
    ap.add_argument("--synthetic-ratio", type=float, default=1.0,
                    help="synthetic examples per real example (keeps real data from being drowned out)")
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--commitpackft", default=None)
    ap.add_argument("--mix", default="0.30,0.25,0.45", help="synthetic clean,decoy,slop")
    ap.add_argument("--test-frac", type=float, default=0.12)
    ap.add_argument("--valid-frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    slop = load_real_slop(a.labeled, a.max_cat_share, a.seed, a.gated_share)
    clean = load_real_clean(a.clean, cap=int(len(slop) * a.clean_ratio) if slop else 10**9)
    n_synth = min(a.synthetic, int(a.synthetic_ratio * (len(slop) + len(clean)))) if (slop or clean) else a.synthetic
    print(f"real clean: {len(clean)}   synthetic: {n_synth}", file=sys.stderr)
    synth = synthetic(a.source, a.commitpackft, n_synth, [float(x) for x in a.mix.split(",")], a.seed) \
        if (a.source or a.commitpackft) and n_synth else []

    splits: dict[str, list] = defaultdict(list)
    for ex in slop + clean:
        b = bucket(ex["repo"])
        splits["test" if b < a.test_frac else "valid" if b < a.test_frac + a.valid_frac else "train"].append(ex)
    for ex in synth:  # synthetic never goes into test
        splits["valid" if bucket(ex["repo"]) < a.valid_frac else "train"].append(ex)

    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {"total": 0}
    for name in ("train", "valid", "test"):
        rows = splits[name]
        rng.shuffle(rows)
        c = Counter()
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(json.dumps({"messages": build_messages(ex["hunk"], ex["findings"]),
                                     "meta": {"source": ex["source"], "repo": ex["repo"], "lang": ex["lang"]}},
                                    ensure_ascii=False) + "\n")
                c["examples"] += 1
                c[f"source:{ex['source']}"] += 1
                c[f"lang:{ex['lang']}"] += 1
                c["no-findings" if not ex["findings"] else "with-findings"] += 1
                for f in ex["findings"]:
                    c[f"cat:{f.category}"] += 1
        stats[name] = dict(sorted(c.items()))
        stats["total"] += c["examples"]
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
