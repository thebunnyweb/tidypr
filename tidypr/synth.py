"""Synthetic training data, v2: realistic hunks, known slop, and look-alike decoys.

v1 taught the model our templates instead of the concept (it flagged plain `if`s and `print`s as
"catch-all" on real diffs). v2 fixes the data, not the model:

* Hunk shapes look like real diffs: partial edits with context lines and surrounding file lines,
  not only whole brand-new functions.
* Decoys: code that *looks* like a category but is fine (specific except + log, a comment that
  explains why, `if arg is None: arg = []`, plain logs, tracked TODOs) labelled `{"findings": []}`.
  They also appear next to real slop, so the model has to point at the right lines.
* Several reasons per category, filled with the actual code, so there is no single sentence to parrot.

Labels stay exact because we still know exactly what we injected.
"""
from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass, field

from .diff import Hunk
from .rules import check_hunk
from .schema import Finding

_ASSIGN = re.compile(r"^(\s*)([a-z_][a-z0-9_]*)\s*=\s*[\w.]+\(.*\)\s*$")
_CALL_STMT = re.compile(r"^(\s*)[\w.]+\(.*\)\s*$")
_RETURN_NAME = re.compile(r"^(\s*)return ([a-z_][a-z0-9_]*)\s*$")
_RETURN = re.compile(r"^(\s*)return\b")
_FOR = re.compile(r"^(\s*)for ([a-z_][a-z0-9_]*) in ([a-z_][a-z0-9_.]*)\s*:\s*$")
_KEYWORDS = {"None", "True", "False", "self", "cls"}
_INDENT = re.compile(r"^(\s*)")


@dataclass
class Func:
    path: str
    start: int            # 1-based line number of `def` in the source file
    lines: list[str]
    name: str
    body_idx: int         # index (into lines) of the first body statement
    has_docstring: bool
    protected: set[int]   # indices inside the docstring; never modified
    before: list[str] = field(default_factory=list)   # up to 3 file lines above the def (context)
    after: list[str] = field(default_factory=list)    # up to 3 file lines below (context)
    none_params: list[str] = field(default_factory=list)  # params whose default is None


def _balanced(s: str) -> bool:
    return s.count("(") == s.count(")") and s.count("[") == s.count("]") and s.count("{") == s.count("}") \
        and s.count('"') % 2 == 0 and s.count("'") % 2 == 0


def human(name: str) -> str:
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ").strip().lower()
    return words or name


def _ind(line: str) -> str:
    return _INDENT.match(line).group(1)


def extract_functions(source: str, path: str, min_lines: int = 5, max_lines: int = 40) -> list[Func]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    src = source.splitlines()
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.body:
            continue
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        n = node.end_lineno - start + 1
        if not (min_lines <= n <= max_lines):
            continue
        lines = src[start - 1: node.end_lineno]
        if any("\t" in l for l in lines):
            continue
        first = node.body[0]
        has_doc = isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
            and isinstance(first.value.value, str)
        protected = set(range(first.lineno - start, first.end_lineno - start + 1)) if has_doc else set()
        body_idx = (node.body[1] if has_doc and len(node.body) > 1 else first).lineno - start
        args = node.args.args + node.args.kwonlyargs
        defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + list(node.args.defaults) \
            + list(node.args.kw_defaults)
        none_params = [a.arg for a, d in zip(args, defaults)
                       if isinstance(d, ast.Constant) and d.value is None]
        out.append(Func(path, start, lines, node.name, body_idx, has_doc, protected,
                        before=src[max(0, start - 4): start - 1], after=src[node.end_lineno: node.end_lineno + 3],
                        none_params=none_params))
    return out


# ---- tagged lines --------------------------------------------------------------------------------
# Each row is [text, tag, kind]: tag = finding id (slop) / "decoy" / None, kind = "+" added or " " context.

def _rows(func: Func, rng: random.Random) -> tuple[list[list], str]:
    """Pick a realistic diff shape for this function."""
    if rng.random() < 0.35:
        shape = "new"                       # brand-new function: every line added
        rows = [[t, None, "+"] for t in func.lines]
    else:
        shape = "edit"                      # change a few lines inside an existing function
        n = len(func.lines)
        lo = max(func.body_idx, 1)
        a = rng.randint(lo, n - 1)
        b = min(n, a + rng.randint(1, 8))
        rows = [[t, None, "+" if a <= i < b else " "] for i, t in enumerate(func.lines)]
    return rows, shape


def _cands(rows, func: Func, pattern: re.Pattern | None = None) -> list[int]:
    out = []
    for i, (t, tag, _) in enumerate(rows):
        if tag is not None or i < func.body_idx or i in func.protected or not t.strip() or not _balanced(t):
            continue
        if pattern is None or pattern.match(t):
            out.append(i)
    return out


# ---- slop injectors: mutate rows, return (category, severity, reason, fix) --------------------------

def inj_redundant_comment(rows, func, rng, tag, shape):
    pick = rng.choice([_ASSIGN, _FOR, _RETURN])
    idx = _cands(rows, func, pick)
    if not idx:
        return None
    i = rng.choice(idx)
    line = rows[i][0]
    if pick is _ASSIGN:
        ind, var = _ASSIGN.match(line).groups()
        text = f"# {rng.choice(['Get the', 'Calculate the', 'Store the', 'Set the', 'Create the', 'Compute the'])} {human(var)}"
    elif pick is _FOR:
        ind, var, seq = _FOR.match(line).groups()
        text = f"# {rng.choice(['Loop over', 'Iterate over', 'Go through each item in', 'For each'])} {human(seq.split('.')[-1])}"
    else:
        ind = _ind(line)
        text = f"# {rng.choice(['Return the result', 'Return the value', 'Return it', 'Finally, return the output'])}"
    rows.insert(i, [ind + text, tag, "+"])
    reason = rng.choice([
        f"`{text}` just restates the next line; delete it.",
        "This comment repeats what the code already says and adds nothing a reader needs.",
        f"Redundant comment: `{line.strip()[:50]}` is self-explanatory.",
        "Comments should explain why, not narrate what the next line does.",
    ])
    return "redundant-comment", "low", reason, ""


def inj_catch_all(rows, func, rng, tag, shape):
    idx = [i for i in _cands(rows, func, _CALL_STMT) if not _RETURN.match(rows[i][0])]
    if not idx:
        return None
    i = rng.choice(idx)
    orig = rows[i][0]
    ind, call = _ind(orig), orig.strip()
    exc = rng.choice(["except Exception:", "except Exception as e:", "except:", "except BaseException:"])
    body = rng.choice(["pass", "pass", "return None"])
    rows[i:i + 1] = [[f"{ind}try:", tag, "+"], [f"{ind}    {call}", tag, "+"], [f"{ind}{exc}", tag, "+"],
                     [f"{ind}    {body}", tag, "+"]]
    short = call[:50]
    reason = rng.choice([
        f"`{exc.rstrip(':')}: {body}` swallows every error from `{short}`, so failures disappear silently.",
        f"If `{short}` fails, nobody will ever know: the catch-all hides it. Catch the specific error or let it raise.",
        f"Catch-all exception handler around `{short}` hides bugs; handle the exceptions you expect and log them.",
    ])
    return "over-defensive", "high", reason, orig


def inj_pointless_reraise(rows, func, rng, tag, shape):
    idx = [i for i in _cands(rows, func, _CALL_STMT) if not _RETURN.match(rows[i][0])]
    if not idx:
        return None
    i = rng.choice(idx)
    orig = rows[i][0]
    ind = _ind(orig)
    rows[i:i + 1] = [[f"{ind}try:", tag, "+"], [f"{ind}    {orig.strip()}", tag, "+"],
                     [f"{ind}except Exception as e:", tag, "+"], [f"{ind}    raise e", tag, "+"]]
    reason = rng.choice([
        "`except Exception as e: raise e` re-raises unchanged; the try/except does nothing but add noise.",
        "This try/except only re-raises the same exception. Remove it.",
    ])
    return "over-defensive", "medium", reason, orig


def inj_none_wrap(rows, func, rng, tag, shape):
    idx = [i for i in _cands(rows, func, _RETURN_NAME) if _RETURN_NAME.match(rows[i][0]).group(2) not in _KEYWORDS]
    if not idx:
        return None
    i = rng.choice(idx)
    orig = rows[i][0]
    ind, var = _RETURN_NAME.match(orig).groups()
    rows[i:i + 1] = [[f"{ind}if {var} is not None:", tag, "+"], [f"{ind}    return {var}", tag, "+"],
                     [f"{ind}else:", tag, "+"], [f"{ind}    return None", tag, "+"]]
    reason = rng.choice([
        f"Both branches return `{var}` (None is None); `return {var}` behaves identically.",
        f"The `if {var} is not None` check changes nothing: the else branch returns the same None.",
        f"Needless if/else around `return {var}`.",
    ])
    return "over-defensive", "medium", reason, orig


_DOC_OPENERS = ["This function is responsible for {h}.", "This method handles the process of {h}.",
                "A robust and comprehensive helper to {h}.", "This function serves as a utility to {h}.",
                "This utility function is designed to {h} in an efficient manner.",
                "Helper method that takes care of {h}."]
_DOC_FILLER = ["It takes the provided inputs and processes them accordingly.",
               "This implementation is designed to be efficient, reliable and maintainable.",
               "It gracefully handles all edge cases to ensure correct behaviour.",
               "The function follows best practices for clean and readable code.",
               "This provides a seamless experience for callers of this API.",
               "It leverages Python's built-in capabilities to achieve the desired outcome.",
               "Overall, this ensures robust and predictable results."]


def _has_doc_at(rows, i) -> bool:
    return i < len(rows) and rows[i][0].strip().startswith(('"""', "'''"))


def inj_verbose_docstring(rows, func, rng, tag, shape):
    if _has_doc_at(rows, func.body_idx):
        return None
    if func.has_docstring or func.body_idx < 1 or not rows[func.body_idx - 1][0].rstrip().endswith(":"):
        return None
    at = func.body_idx
    ind = _ind(rows[at][0])
    h = human(func.name)
    body = [rng.choice(_DOC_OPENERS).format(h=h)] + rng.sample(_DOC_FILLER, k=rng.randint(2, 4))
    tail = rng.choice([[f"{ind}", f"{ind}Returns:", f"{ind}    The result."], [f"{ind}", f"{ind}Args:",
                       f"{ind}    The input parameters."], []])
    doc = [f'{ind}"""'] + [f"{ind}{s}" for s in body] + tail + [f'{ind}"""']
    rows[at:at] = [[d, tag, "+"] for d in doc]
    reason = rng.choice([
        f"{len(doc)}-line docstring of filler that never says what `{func.name}` returns or does.",
        "Docstring is marketing copy ('robust', 'seamless', 'best practices'); replace with one factual line.",
        "Verbose docstring with no information; a one-line summary is enough.",
    ])
    return "verbose-docstring", "low", reason, f'{ind}"""{h.capitalize()}."""'


def inj_emoji_log(rows, func, rng, tag, shape):
    idx = _cands(rows, func, _RETURN) or _cands(rows, func, _CALL_STMT)
    if not idx:
        return None
    i = rng.choice(idx)
    ind = _ind(rows[i][0])
    h = human(func.name).capitalize()
    msg = rng.choice([f"✅ {h} completed successfully! 🎉", f"🚀 Starting {h.lower()}...", f"✨ {h} done!",
                      f"🔥 {h} finished", f"🎯 {h} worked perfectly!", f"👍 All good: {h.lower()}"])
    call = rng.choice(["logger.info", "logging.info", "print", "log.debug"])
    rows.insert(i, [f'{ind}{call}("{msg}")', tag, "+"])
    reason = rng.choice([
        "Emoji and celebration in a log line make logs noisy and hard to grep.",
        f"`{call}(\"{msg[:25]}...\")` is decoration, not information.",
        "Log noise: drop it or log something useful (ids, counts) without emoji.",
    ])
    return "log-noise", "low", reason, ""


def inj_todo(rows, func, rng, tag, shape):
    idx = _cands(rows, func)
    if not idx:
        return None
    i = rng.choice(idx)
    msg = rng.choice(["# TODO: implement proper error handling", "# TODO: add validation here",
                      "# TODO: handle edge cases", "# TODO: replace with real implementation",
                      "# TODO: implement this properly", "# TODO: add proper logging"])
    rows.insert(i, [_ind(rows[i][0]) + msg, tag, "+"])
    reason = rng.choice([
        "Placeholder TODO in new code with no owner or issue; do it now or track it.",
        f"`{msg[2:]}` is a vague placeholder that will never get done.",
    ])
    return "dead-or-placeholder", "medium", reason, ""


SLOP = [inj_redundant_comment, inj_catch_all, inj_pointless_reraise, inj_none_wrap,
        inj_verbose_docstring, inj_emoji_log, inj_todo]
_SLOP_WEIGHTS = [1.3, 1.0, 0.5, 0.8, 1.0, 1.0, 0.6]


# ---- decoys: look like a category, but are fine. Tagged "decoy", never labelled ----------------------

def dec_specific_except(rows, func, rng, shape):
    idx = [i for i in _cands(rows, func, _CALL_STMT) if not _RETURN.match(rows[i][0])]
    if not idx:
        return False
    i = rng.choice(idx)
    orig = rows[i][0]
    ind = _ind(orig)
    exc = rng.choice(["OSError", "KeyError", "ValueError", "(ValueError, TypeError)", "FileNotFoundError",
                      "json.JSONDecodeError", "TimeoutError"])
    handler = rng.choice([
        [f'{ind}    logger.warning("{human(func.name)} failed: %s", exc)', f"{ind}    raise"],
        [f'{ind}    logger.exception("{human(func.name)} failed")', f"{ind}    raise"],
        [f"{ind}    raise RuntimeError(\"{human(func.name)} failed\") from exc"],
        [f'{ind}    log.debug("skipping: %s", exc)', f"{ind}    return None"],
    ])
    rows[i:i + 1] = [[f"{ind}try:", "decoy", "+"], [f"{ind}    {orig.strip()}", "decoy", "+"],
                     [f"{ind}except {exc} as exc:", "decoy", "+"]] + [[h, "decoy", "+"] for h in handler]
    return True


_WHY = ["# NOTE: order matters here; callers rely on insertion order.",
        "# Avoid a second lookup on the hot path.",
        "# Kept for backwards compatibility with older callers.",
        "# Windows returns bytes here, so normalise first.",
        "# Must run before the lock is released.",
        "# The upstream API caps this at 100 per page.",
        "# Copy so callers can't mutate our cached value.",
        "# See RFC 3986 section 5.2 for the merge rules."]


def dec_why_comment(rows, func, rng, shape):
    idx = _cands(rows, func)
    if not idx:
        return False
    i = rng.choice(idx)
    rows.insert(i, [_ind(rows[i][0]) + rng.choice(_WHY), "decoy", "+"])
    return True


def dec_none_default(rows, func, rng, shape):
    if not func.none_params or func.body_idx >= len(rows):
        return False
    p = rng.choice(func.none_params)
    ind = _ind(rows[func.body_idx][0])
    default = rng.choice(["[]", "{}", "set()", "{}"])
    rows[func.body_idx:func.body_idx] = [[f"{ind}if {p} is None:", "decoy", "+"],
                                         [f"{ind}    {p} = {default}", "decoy", "+"]]
    return True


def dec_plain_log(rows, func, rng, shape):
    idx = _cands(rows, func, _RETURN_NAME) or _cands(rows, func, _RETURN)
    if not idx:
        return False
    i = rng.choice(idx)
    m = _RETURN_NAME.match(rows[i][0])
    var = m.group(2) if m else None
    msg = (f'logger.debug("{human(func.name)}: %r", {var})' if var and var not in _KEYWORDS
           else f'logger.debug("{human(func.name)} finished")')
    rows.insert(i, [_ind(rows[i][0]) + msg, "decoy", "+"])
    return True


def dec_tracked_todo(rows, func, rng, shape):
    idx = _cands(rows, func)
    if not idx:
        return False
    i = rng.choice(idx)
    n = rng.randint(100, 9999)
    msg = rng.choice([f"# TODO(#{n}): remove once Python 3.9 support is dropped.",
                      f"# FIXME(#{n}): upstream bug, drop this workaround after the next release.",
                      f"# TODO(gh-{n}): switch to the async client when it's stable."])
    rows.insert(i, [_ind(rows[i][0]) + msg, "decoy", "+"])
    return True


def dec_short_docstring(rows, func, rng, shape):
    if _has_doc_at(rows, func.body_idx):
        return False
    if func.has_docstring or func.body_idx < 1 or not rows[func.body_idx - 1][0].rstrip().endswith(":"):
        return False
    ind = _ind(rows[func.body_idx][0])
    rows.insert(func.body_idx, [f'{ind}"""{human(func.name).capitalize()}."""', "decoy", "+"])
    return True


DECOYS = [dec_specific_except, dec_why_comment, dec_none_default, dec_plain_log, dec_tracked_todo,
          dec_short_docstring]


# ---- assembling examples -------------------------------------------------------------------------------

def to_hunk(func: Func, rows: list[list], with_context: bool = True) -> Hunk:
    ctx_b = func.before if with_context else []
    ctx_a = func.after if with_context else []
    allrows = [[t, None, " "] for t in ctx_b] + rows + [[t, None, " "] for t in ctx_a]
    first = func.start - len(ctx_b)
    return Hunk(func.path, [(first + i, k, t) for i, (t, _, k) in enumerate(allrows)])


def make_example(func: Func, rng: random.Random, kind: str, max_slop: int = 2,
                 decoy_in_slop: float = 0.35) -> tuple[Hunk, list[Finding]] | None:
    """kind: "clean" (untouched), "decoy" (look-alikes only), "slop" (1..max_slop injections, maybe decoys)."""
    rows, shape = _rows(func, rng)
    with_ctx = rng.random() < 0.7
    if check_hunk(to_hunk(func, rows, with_ctx)):
        return None  # base code already trips a rule: not a trustworthy "clean" baseline

    specs: dict[int, tuple] = {}
    if kind in ("decoy", "slop"):
        want_decoys = 1 if kind == "slop" else rng.randint(1, 2)
        if kind == "decoy" or rng.random() < decoy_in_slop:
            for d in rng.sample(DECOYS, k=len(DECOYS)):
                if want_decoys <= 0:
                    break
                want_decoys -= d(rows, func, rng, shape)
            if kind == "decoy" and want_decoys > 0 and not any(r[1] == "decoy" for r in rows):
                return None
    if kind == "slop":
        want = rng.randint(1, max_slop)
        order = rng.choices(SLOP, weights=_SLOP_WEIGHTS, k=len(SLOP) * 2)
        for inj in order:
            if len(specs) >= want:
                break
            if inj in [s[4] for s in specs.values()]:
                continue
            tag = len(specs)
            spec = inj(rows, func, rng, tag, shape)
            if spec:
                specs[tag] = (*spec, inj)
        if not specs:
            return None

    hunk = to_hunk(func, rows, with_ctx)
    if kind in ("clean", "decoy") and check_hunk(hunk):
        return None  # a decoy must not look like slop to the rules either
    offset = len(func.before) if with_ctx else 0
    findings = []
    for tag, (cat, sev, reason, fix, _) in specs.items():
        idx = [i for i, r in enumerate(rows) if r[1] == tag]
        first, last = hunk.lines[offset + idx[0]][0], hunk.lines[offset + idx[-1]][0]
        findings.append(Finding(func.path, first, cat, sev, reason, end_line=last, fix=fix, source="synth"))
    findings.sort(key=lambda f: f.line)
    return hunk, findings


# v1 API, kept for the tests and older scripts
def clean_example(func: Func) -> tuple[Hunk, list[Finding]] | None:
    hunk = to_hunk(func, [[t, None, "+"] for t in func.lines], with_context=False)
    return None if check_hunk(hunk) else (hunk, [])


def slop_example(func: Func, rng: random.Random, max_injections: int = 2) -> tuple[Hunk, list[Finding]] | None:
    return make_example(func, rng, "slop", max_slop=max_injections)
