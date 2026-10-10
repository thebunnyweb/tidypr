"""Zero-dependency rule pass. Cheap, precise, and the baseline the model must beat."""
from __future__ import annotations

import ast
import re

from .diff import Hunk
from .schema import Finding

_CATCH_ALL = re.compile(r"^\s*except\s*(\(?\s*(Base)?Exception\s*\)?)?\s*(as\s+\w+)?\s*:\s*$")
_SWALLOW = re.compile(r"^\s*(pass|continue|\.\.\.|return None|return)\s*(#.*)?$")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿✅❌]")
_LOG_CALL = re.compile(r"\b(log(ger)?|logging)\.(debug|info|warning|error|exception|critical)\(|\bprint\(")
_TODO = re.compile(r"#\s*(TODO|FIXME)\b[:\s]*(implement|add|handle|placeholder|replace)", re.I)
_FILLER = re.compile(
    r"\b(this (function|method|class) is responsible for|robust and comprehensive|comprehensive solution|"
    r"seamlessly|handles all edge cases|gracefully handles|elegant(ly)? )", re.I)
_COMMENT = re.compile(r"^\s*#\s*(.+?)\s*$")
_WORD = re.compile(r"[a-zA-Z]+")
_STOP = {"the", "a", "an", "to", "of", "if", "is", "it", "and", "we", "this", "that", "from", "for",
         "in", "on", "by", "with", "be", "not", "get", "set", "check", "now", "then", "here", "value"}


def _added(hunk: Hunk) -> list[tuple[int, str]]:
    return [(n, t) for n, k, t in hunk.lines if k == "+"]


def _words(s: str) -> set[str]:
    out = set()
    for w in _WORD.findall(s):
        for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", w) or [w]:
            out.add(part.lower())
    return out - _STOP


def check_hunk(hunk: Hunk) -> list[Finding]:
    path, added = hunk.path, _added(hunk)
    by_line = dict(added)
    out: list[Finding] = []

    for n, text in added:
        # over-defensive: catch-all except that swallows
        if _CATCH_ALL.match(text) and _SWALLOW.match(by_line.get(n + 1, "")):
            out.append(Finding(path, n, "over-defensive", "high",
                               "Catch-all except silently swallows every error, so failures go unnoticed. "
                               "Catch the specific exception and handle or log it.",
                               end_line=n + 1, source="rules"))
        # log-noise: emoji in log/print
        if _LOG_CALL.search(text) and _EMOJI.search(text):
            out.append(Finding(path, n, "log-noise", "low",
                               "Emoji/celebratory text in a log message adds noise to logs.",
                               fix=_strip_emoji(text), source="rules"))
        # placeholder TODOs
        if _TODO.search(text):
            out.append(Finding(path, n, "dead-or-placeholder", "medium",
                               "Placeholder TODO left in new code; implement it or track it in an issue.",
                               source="rules"))
        # docstring / comment filler phrases (one finding per block, not per line)
        if _FILLER.search(text) and not any(f.category == "verbose-docstring" and n - f.line <= 6 for f in out):
            out.append(Finding(path, n, "verbose-docstring", "low",
                               "Filler wording ('robust', 'comprehensive', 'responsible for') says nothing "
                               "about behaviour; state what it returns in one line.", source="rules"))

    # redundant-comment: comment whose meaningful words all appear in the next code line
    for n, text in added:
        m = _COMMENT.match(text)
        nxt = by_line.get(n + 1)
        if not m or nxt is None or _COMMENT.match(nxt) or not nxt.strip():
            continue
        cw = _words(m.group(1))
        if 1 <= len(cw) <= 6 and cw <= _words(nxt):
            out.append(Finding(path, n, "redundant-comment", "low",
                               "Comment restates the next line of code.", fix="", source="rules"))

    # commented-out code: 3+ consecutive added comment lines that parse as Python
    run: list[tuple[int, str]] = []
    for n, text in added + [(-1, "")]:
        m = _COMMENT.match(text)
        body = m.group(1) if m else None
        if body and _looks_like_code(body) and (not run or run[-1][0] == n - 1):
            run.append((n, body))
            continue
        if len(run) >= 3:
            out.append(Finding(path, run[0][0], "dead-or-placeholder", "medium",
                               "Commented-out code block; delete it (git keeps history).",
                               end_line=run[-1][0], fix="", source="rules"))
        run = [(n, body)] if body and _looks_like_code(body) else []
    return out


def _strip_emoji(line: str) -> str:
    line = re.sub(r"\s*(?:" + _EMOJI.pattern + r")+\s*", " ", line)
    return re.sub(r"([\"'])\s*(.*?)\s*\1", lambda m: f"{m[1]}{m[2].strip()}{m[1]}", line).rstrip()


def _looks_like_code(s: str) -> bool:
    if not re.search(r"[=()\[\]:]", s) or s.endswith("."):
        return False
    try:
        ast.parse(s.strip().rstrip(":") + ("" if not s.rstrip().endswith(":") else ": pass"))
        return True
    except SyntaxError:
        return False
