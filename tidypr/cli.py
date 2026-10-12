"""tidypr CLI.

  tidypr review --diff pr.diff                 # rules only
  git diff main... | tidypr review --diff -    # from stdin
  tidypr review --diff pr.diff --backend llamacpp --model bunnyweb/tidypr-1.5b-gguf
  tidypr review --diff pr.diff --backend mlx --model Qwen/Qwen2.5-Coder-0.5B-Instruct --adapter runs/mlx/adapters
  tidypr github                                # inside GitHub Actions
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .diff import code_hunks, parse_diff
from .infer import DEFAULT_CATEGORIES, ReviewConfig, findings_json, make_backend, review_hunks
from .schema import CATEGORIES
from .schema import SEVERITIES, severity_at_least


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["rules", "llamacpp", "mlx"], default=os.environ.get("TIDYPR_BACKEND", "rules"))
    p.add_argument("--model", default=os.environ.get("TIDYPR_MODEL") or None,
                   help="HF repo id, local .gguf, or local MLX dir")
    p.add_argument("--model-file", default=os.environ.get("TIDYPR_MODEL_FILE", "*Q8_0.gguf"),
                   help="GGUF filename/glob inside the HF repo (llamacpp)")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir (mlx)")
    p.add_argument("--no-rules", action="store_true")
    p.add_argument("--min-severity", choices=SEVERITIES, default="low")
    p.add_argument("--max-findings", type=int, default=10)
    p.add_argument("--include-tests", action="store_true", help="also review tests/ and test_*.py files")
    p.add_argument("--categories", default="default",
                   help="comma-separated categories to report, 'default' (reliable ones) or 'all'")


def _cfg(a) -> ReviewConfig:
    cats = {"default": DEFAULT_CATEGORIES, "all": frozenset(CATEGORIES)}.get(
        a.categories, frozenset(c.strip() for c in a.categories.split(",") if c.strip()))
    return ReviewConfig(a.min_severity, a.max_findings, not a.no_rules, cats)


def _backend(a):
    return make_backend(a.backend, a.model, filename=a.model_file, adapter_path=a.adapter)


def cmd_review(a) -> int:
    text = sys.stdin.read() if a.diff == "-" else open(a.diff, encoding="utf-8").read()
    hunks = code_hunks(parse_diff(text), skip_tests=not a.include_tests)
    findings = review_hunks(hunks, _backend(a), _cfg(a))
    if a.format == "json":
        print(findings_json(findings))
    else:
        for f in findings:
            print(f"{f.path}:{f.span[0]}-{f.span[1]}  [{f.severity}] {f.category} ({f.source})\n    {f.reason}")
            if f.fix is not None:
                print("    fix: " + (f.fix.replace("\n", "\n         ") or "<delete lines>"))
        print(f"\n{len(findings)} finding(s) in {len(hunks)} hunk(s).")
    return 1 if a.fail_on and any(severity_at_least(f.severity, a.fail_on) for f in findings) else 0


def cmd_github(a) -> int:
    from .github import GitHub, post

    event = json.load(open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8"))
    pr = event.get("pull_request")
    if not pr:
        print("tidypr: not a pull_request event, skipping.")
        return 0
    token = os.environ.get("TIDYPR_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    gh = GitHub(os.environ["GITHUB_REPOSITORY"], token)
    hunks = code_hunks(parse_diff(gh.pr_diff(pr["number"])), skip_tests=not a.include_tests)
    findings = review_hunks(hunks, _backend(a), _cfg(a))

    code_of = {(h.path, n): t for h in hunks for n, _, t in h.lines}
    added = sum(len(h.added_lines) for h in hunks)
    used = post(gh, pr["number"], pr["head"]["sha"], findings, code_of, added,
                mode=a.post_mode, min_inline=a.min_inline, model=a.model)
    print(f"tidypr: {len(findings)} finding(s), posted via {used}.")

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"count={len(findings)}\n")
    if a.fail_on and any(severity_at_least(f.severity, a.fail_on) for f in findings):
        print(f"::error::tidypr found {a.fail_on}+ severity findings")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tidypr")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("review", help="review a unified diff locally")
    r.add_argument("--diff", required=True, help="diff file, or - for stdin")
    r.add_argument("--format", choices=["text", "json"], default="text")
    r.add_argument("--fail-on", choices=SEVERITIES, default=None)
    _add_model_args(r)

    g = sub.add_parser("github", help="review the current PR inside GitHub Actions")
    g.add_argument("--post-mode", choices=["summary-only", "review", "annotations"], default="summary-only")
    g.add_argument("--min-inline", choices=SEVERITIES, default="medium")
    g.add_argument("--fail-on", choices=SEVERITIES, default=None)
    _add_model_args(g)

    a = ap.parse_args(argv)
    if getattr(a, "fail_on", None) == "":
        a.fail_on = None
    return {"review": cmd_review, "github": cmd_github}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
