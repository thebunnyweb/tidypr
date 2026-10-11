"""Language-agnostic sanity checks: a label must match what is actually on those lines.

Used three times: on teacher labels, when merging the dataset, and on the model's own findings at review time.
"""
from __future__ import annotations

import re

_COMMENT_LINE = re.compile(r"^\s*(#|//|/\*|\*|<!--|--|;|\"\"\"|\'\'\')|\s(#|//)\s")
_LOG_CALL = re.compile(r"console\.(log|debug|info|warn|error)|\bprint(ln|f)?\s*\(|\blog(ger|ging)?\.\w+\(|"
                       r"fmt\.Print|System\.(out|err)\.print|\bputs\b|\becho\b|\bdebug!?\(|NSLog|Log\.[dive]\(", re.I)
_TODO = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b", re.I)
_DEFENSIVE = re.compile(r"\b(try|catch|except|rescue|finally|if|null|None|nil|err|undefined|isinstance|typeof|"
                        r"Optional|guard|unless)\b|\?\?|\?\.|!= ?nil|is not None", re.I)


def consistent(cat: str, reason: str, lines: list[str]) -> bool:
    text = "\n".join(lines)
    if "todo" in reason.lower() and not _TODO.search(text):
        return False                      # reason talks about a TODO that isn't there
    if cat == "redundant-comment":
        return any(_COMMENT_LINE.search(l) for l in lines)
    if cat == "log-noise":
        return bool(_LOG_CALL.search(text))
    if cat == "over-defensive":
        return bool(_DEFENSIVE.search(text))
    if cat == "verbose-docstring":
        return any(_COMMENT_LINE.search(l) for l in lines)
    return True
