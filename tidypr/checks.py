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


# Reasons that say which category a finding really is, whatever label it was given.
_DEAD = re.compile(r"\bunused\b|never used|not used|isn.t used|dead code|commented.out|leftover|left over", re.I)
_ABOUT_COMMENT = re.compile(r"comment|docstring|doc ?block|jsdoc|javadoc|narrat|restat|explain|describ|says\b", re.I)
_ABOUT_CHECK = re.compile(r"condition|check|null|none\b|nil\b|guard|already|always|never be|try|catch|except", re.I)
_TOO_THIN = re.compile(r"lacks|too short|missing|insufficient|unclear|not enough", re.I)
_TOO_MUCH = re.compile(r"verbose|long|wordy|repeat|restat|redundant|obvious|narrat", re.I)
_NOT_VISIBLE = re.compile(r"another commit|other commit|unrelated|different pr|separate pr|out of scope", re.I)


def reconcile(cat: str, reason: str) -> str | None:
    """Fix a category that contradicts its own reason; None means the label is unusable.

    The teacher often names the right problem but files it under the wrong category
    ("Import is unused." as needless-abstraction). Applied to teacher labels and when merging.
    """
    if _NOT_VISIBLE.search(reason):
        return None                       # not something one hunk can show
    if cat != "dead-or-placeholder" and cat != "log-noise" and _DEAD.search(reason) \
            and not _ABOUT_COMMENT.search(reason):
        return "dead-or-placeholder"
    if cat == "redundant-comment" and not _ABOUT_COMMENT.search(reason):
        return "over-defensive" if _ABOUT_CHECK.search(reason) else None
    if cat == "verbose-docstring" and _TOO_THIN.search(reason) and not _TOO_MUCH.search(reason):
        return None                       # "too short" is the opposite of this category
    return cat
