"""Prompt format. Training and inference MUST both go through these functions."""
from __future__ import annotations

import json

from .diff import Hunk
from .langs import language_of
from .schema import CATEGORIES, Finding

SYSTEM_PROMPT = (
    "You review pull-request hunks (any programming language) for AI slop: code that is bloated, needlessly defensive, "
    "or filler. Flag only clear problems on added lines (marked '+'). Ignore style that a linter "
    "would catch and never comment on correct, idiomatic code.\n\nCategories:\n"
    + "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())
    + "\n\nSeverity: high = hides bugs or breaks behaviour; medium = real maintenance cost; "
    "low = noise.\nReply with JSON only: "
    '{"findings": [{"line": int, "end_line": int, "category": str, "severity": str, '
    '"reason": str, "fix": str (optional replacement code for line..end_line)}]}. '
    'Use {"findings": []} when the hunk is fine.'
)


def user_message(hunk: Hunk) -> str:
    lang = language_of(hunk.path)
    head = f"File: {hunk.path}" + (f"\nLanguage: {lang}" if lang else "")
    return f"{head}\n```diff\n{hunk.render()}\n```"


def build_messages(hunk: Hunk, findings: list[Finding] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(hunk)}]
    if findings is not None:
        msgs.append({"role": "assistant", "content": target_json(findings)})
    return msgs


def target_json(findings: list[Finding]) -> str:
    return json.dumps({"findings": [f.to_target() for f in findings]}, ensure_ascii=False)
