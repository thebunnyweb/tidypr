"""Collect real (hunk, human review comment) pairs from public datasets and keep the slop-sounding ones.

  python data/real/collect.py --out data/real/candidates.jsonl --peek        # check a few rows first
  python data/real/collect.py --out data/real/candidates.jsonl \
      --aidev 6000 --codereviewer 6000

Sources:
  aidev         hao-li/AIDev pr_review_comments: humans reviewing PRs written by AI agents (best fit)
  codereviewer  fasterinnerlooper/codereviewer (Microsoft CodeReviewer): 9 languages, human reviews
  gcr           ronantakizawa/github-codereview: off by default until its diff_context format is checked

Each output line: {id, source, repo, path, lang, hunk: [[line, kind, text], ...], anchor, comment, score}
`anchor` is the new-file line the reviewer commented on (None when the source doesn't say).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tidypr.diff import Hunk, is_test, parse_diff  # noqa: E402
from tidypr.langs import LANGS, language_of, normalise  # noqa: E402

_EXT_OF = {}
for _ext, _lang in LANGS.items():
    _EXT_OF.setdefault(_lang, _ext)

# Phrases reviewers use when they point at slop. Weighted: strong phrases count more.
KEYWORDS: dict[str, float] = {
    # over-defensive
    "swallow": 3, "silently": 2, "catch-all": 3, "catch all": 2, "bare except": 3, "broad except": 3,
    "too broad": 2, "empty catch": 3, "unnecessary try": 3, "why catch": 2, "why the try": 2,
    "unnecessary check": 3, "redundant check": 3, "can never be null": 3, "can't be none": 3, "cannot be none": 3,
    "can't be null": 3, "cannot be null": 3, "never be none": 3, "never be null": 3, "defensive": 2, "overkill": 2,
    "already checked": 2, "already validated": 2, "not needed": 2, "unnecessary null": 3, "is always": 1,
    # redundant-comment
    "redundant comment": 3, "unnecessary comment": 3, "obvious comment": 3, "remove this comment": 3,
    "remove the comment": 3, "comment is redundant": 3, "comment doesn't add": 3, "comments are redundant": 3,
    "self-explanatory": 2, "self explanatory": 2, "comment adds nothing": 3, "too many comments": 3,
    # verbose-docstring
    "docstring": 1.5, "too verbose": 3, "verbose": 2, "wordy": 2, "jsdoc": 1, "javadoc": 1,
    # needless-abstraction
    "over-engineer": 3, "overengineer": 3, "over engineer": 3, "unnecessary abstraction": 3, "yagni": 3,
    "unnecessary wrapper": 3, "unnecessary class": 3, "unnecessary helper": 3, "just inline": 2,
    "inline this": 2, "too complex": 1.5, "simplify": 1, "indirection": 2,
    # dead-or-placeholder
    "dead code": 3, "commented out": 3, "commented-out": 3, "unused": 2, "leftover": 2, "left over": 2,
    "placeholder": 2, "stub": 1.5, "todo": 1.5, "not used": 2, "never used": 2,
    # log-noise
    "emoji": 3, "console.log": 3, "print statement": 3, "debug print": 3, "remove print": 3, "remove the print": 3,
    "debug log": 2, "noisy": 2, "log spam": 3, "too much logging": 3,
    # hallucinated-api
    "doesn't exist": 2, "does not exist": 2, "no such method": 3, "not a valid": 1.5, "hallucinat": 3,
    "there is no such": 3, "not a real": 2,
    # generic
    "unnecessary": 2, "redundant": 2, "not necessary": 2, "can be removed": 2, "should be removed": 2,
    "remove this": 1, "why is this needed": 2, "is this needed": 2, "do we need": 1.5, "duplicate": 1,
}
_KW = sorted(KEYWORDS, key=len, reverse=True)


def slop_score(comment: str) -> float:
    c = comment.lower()
    return sum(KEYWORDS[k] for k in _KW if k in c)


def hunk_from_diff_hunk(path: str, diff_hunk: str) -> Hunk | None:
    """GitHub review `diff_hunk`: starts at an @@ header and ends at the commented line."""
    text = f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n{diff_hunk.rstrip()}\n"
    hunks = parse_diff(text)
    return hunks[0] if hunks else None


def _last_kind(diff_hunk: str) -> str:
    last = diff_hunk.rstrip("\n").split("\n")[-1]
    return last[:1] if last[:1] in "+- " else " "


def _keep(path: str, comment: str, min_score: float) -> bool:
    return bool(path) and language_of(path) is not None and not is_test(path) \
        and 15 <= len(comment) <= 1200 and slop_score(comment) >= min_score


def _row(source, repo, path, lang, hunk: Hunk, anchor, comment, pr: str | None = None) -> dict:
    hid = hashlib.sha1(f"{source}|{repo}|{path}|{hunk.render()}|{comment}".encode()).hexdigest()[:16]
    return {"id": hid, "source": source, "repo": repo, "pr": pr, "path": path, "lang": lang,
            "hunk": [list(x) for x in hunk.lines], "anchor": anchor, "comment": comment.strip(),
            "score": slop_score(comment)}


# ---- AIDev ------------------------------------------------------------------------------------------

def iter_aidev(min_score: float, max_hunk_lines: int = 80):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("hao-li/AIDev", "pr_review_comments.parquet", repo_type="dataset")
    df = pd.read_parquet(p, columns=["user_type", "diff_hunk", "path", "body", "pull_request_url", "in_reply_to_id"])
    df = df[(df["user_type"].astype(str).str.lower() == "user") & df["in_reply_to_id"].isna()]
    for r in df.itertuples(index=False):
        path, body, dh = str(r.path or ""), str(r.body or ""), str(r.diff_hunk or "")
        if not dh.startswith("@@") or not _keep(path, body, min_score) or _last_kind(dh) == "-":
            continue
        hunk = hunk_from_diff_hunk(path, dh)
        if not hunk or len(hunk.lines) > max_hunk_lines:
            continue
        m = re.search(r"/repos/([^/]+/[^/]+)/pulls/", str(r.pull_request_url))
        yield _row("aidev", m.group(1) if m else "?", path, language_of(path), hunk, hunk.lines[-1][0], body,
                   pr=str(r.pull_request_url))


# ---- CodeReviewer -------------------------------------------------------------------------------------

def iter_codereviewer(min_score: float, files=("generation/gen-train.jsonl", "generation/gen-valid.jsonl"),
                      max_hunk_lines: int = 80):
    from huggingface_hub import hf_hub_download

    for fname in files:
        try:
            p = hf_hub_download("fasterinnerlooper/codereviewer", fname, repo_type="dataset")
        except Exception as e:
            print(f"!! codereviewer {fname}: {type(e).__name__}: {e}")
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                patch, msg = str(r.get("patch") or ""), str(r.get("msg") or "")
                lang = normalise(r.get("lang"))
                # CodeReviewer has no file path; synthesise one so language_of() works downstream
                path = f"src/file{_EXT_OF.get(lang, '.txt')}"
                if not patch.lstrip().startswith("@@") or not _keep(path, msg, min_score):
                    continue
                hunk = hunk_from_diff_hunk(path, patch)
                if not hunk or len(hunk.lines) > max_hunk_lines:
                    continue
                yield _row("codereviewer", str(r.get("proj") or "?"), path, lang, hunk, None, msg)


def iter_codereviewer_clean(max_hunk_lines: int = 60):
    """CodeReviewer quality-estimation files: y == 0 means reviewers accepted the hunk without comment."""
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files("fasterinnerlooper/codereviewer", repo_type="dataset")
             if f.endswith(".jsonl") and ("cls" in f.lower() or "quality" in f.lower())]
    files.sort(key=lambda f: ("train" not in f, f))
    print(f"codereviewer clean: using {files}")
    for fname in files:
        p = hf_hub_download("fasterinnerlooper/codereviewer", fname, repo_type="dataset")
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(r.get("y")) != "0":
                    continue
                patch, lang = str(r.get("patch") or ""), normalise(r.get("lang"))
                path = f"src/file{_EXT_OF.get(lang, '.txt')}"
                if not patch.lstrip().startswith("@@") or language_of(path) is None:
                    continue
                hunk = hunk_from_diff_hunk(path, patch)
                if not hunk or len(hunk.lines) > max_hunk_lines or len(hunk.added_lines) < 2:
                    continue
                yield _row("codereviewer-clean", str(r.get("proj") or "?"), path, lang, hunk, None, "")


# ---- main -------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real/candidates.jsonl")
    ap.add_argument("--aidev", type=int, default=6000, help="max candidates from AIDev (0 = skip)")
    ap.add_argument("--codereviewer", type=int, default=6000, help="max candidates from CodeReviewer (0 = skip)")
    ap.add_argument("--clean", type=int, default=4000, help="max real clean hunks (CodeReviewer y=0) (0 = skip)")
    ap.add_argument("--clean-out", default="data/real/clean.jsonl")
    ap.add_argument("--min-score", type=float, default=2.0, help="keyword score needed to be a candidate")
    ap.add_argument("--peek", action="store_true", help="print 3 candidates per source and stop")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen, total = set(), {}
    with out.open("w", encoding="utf-8") as fh:
        for name, cap, it in [("aidev", a.aidev, lambda: iter_aidev(a.min_score)),
                              ("codereviewer", a.codereviewer, lambda: iter_codereviewer(a.min_score))]:
            if cap <= 0:
                continue
            rows = []
            try:
                for row in it():
                    if row["id"] in seen:
                        continue
                    seen.add(row["id"])
                    rows.append(row)
                    if a.peek and len(rows) >= 3:
                        break
                    if len(rows) >= cap * 3:   # collect extra, keep the highest-scoring `cap`
                        break
            except Exception as e:  # keep going with the other sources
                print(f"!! {name}: {type(e).__name__}: {e}")
            rows.sort(key=lambda r: -r["score"])
            rows = rows[:cap]
            total[name] = len(rows)
            if a.peek:
                for r in rows:
                    print(f"\n===== {name} | {r['repo']} | {r['path']} | anchor={r['anchor']} | score={r['score']}")
                    print(Hunk(r["path"], [tuple(x) for x in r["hunk"]]).render())
                    print(f"--- reviewer: {r['comment'][:400]}")
                continue
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.clean > 0:
        n = 0
        try:
            with Path(a.clean_out).open("w", encoding="utf-8") as fh:
                for r in iter_codereviewer_clean():
                    if a.peek and n >= 3:
                        break
                    if a.peek:
                        print(f"\n===== clean | {r['repo']} | {r['lang']}\n" + Hunk(r["path"], [tuple(x) for x in r["hunk"]]).render())
                    else:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
                    if n >= a.clean:
                        break
        except Exception as e:
            print(f"!! codereviewer clean: {type(e).__name__}: {e}")
        total["clean"] = n
    print(json.dumps({"candidates": total, "out": str(out), "clean_out": a.clean_out}, indent=2))


if __name__ == "__main__":
    main()
