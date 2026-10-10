"""Free synthetic training data: take clean human functions and inject *known* slop.

Because we inject it, labels and fixes are exact. This is the v0 bootstrap; real slop from
AIDev reviews and the hand-labelled gold set come next (see docs/PLAN.md §4).
"""
from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass

from .diff import Hunk
from .rules import check_hunk
from .schema import Finding

_ASSIGN = re.compile(r"^(\s*)([a-z_][a-z0-9_]*)\s*=\s*[\w.]+\(.*\)\s*$")
_CALL_STMT = re.compile(r"^(\s*)[\w.]+\(.*\)\s*$")
_RETURN_NAME = re.compile(r"^(\s*)return ([a-z_][a-z0-9_]*)\s*$")
_RETURN = re.compile(r"^(\s*)return\b")
_KEYWORDS = {"None", "True", "False", "self", "cls"}


@dataclass
class Func:
    path: str
    start: int            # 1-based line number of `def` in the source file
    lines: list[str]
    name: str
    body_idx: int         # index (into lines) of the first body statement
    has_docstring: bool
    protected: set[int]   # indices inside the docstring; never modified


def _balanced(s: str) -> bool:
    return s.count("(") == s.count(")") and s.count("[") == s.count("]") and s.count("{") == s.count("}") \
        and s.count('"') % 2 == 0 and s.count("'") % 2 == 0


def human(name: str) -> str:
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ").strip().lower()
    return words or name


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
        n = node.end_lineno - node.lineno + 1
        if not (min_lines <= n <= max_lines):
            continue
        lines = src[node.lineno - 1: node.end_lineno]
        if any("\t" in l for l in lines):
            continue
        first = node.body[0]
        has_doc = isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
            and isinstance(first.value.value, str)
        protected = set(range(first.lineno - node.lineno, first.end_lineno - node.lineno + 1)) if has_doc else set()
        body_idx = (node.body[1] if has_doc and len(node.body) > 1 else first).lineno - node.lineno
        out.append(Func(path, node.lineno, lines, node.name, body_idx, has_doc, protected))
    return out


# ---- injectors: each mutates `tl` (list of [text, tag]) and returns a finding spec or None ----------

def _candidates(tl, func: Func, pattern: re.Pattern) -> list[int]:
    return [i for i, (t, tag) in enumerate(tl)
            if tag is None and i >= func.body_idx and i not in func.protected and pattern.match(t) and _balanced(t)]


def inj_redundant_comment(tl, func, rng, tag):
    idx = _candidates(tl, func, _ASSIGN)
    if not idx:
        return None
    i = rng.choice(idx)
    ind, var = _ASSIGN.match(tl[i][0]).groups()
    verb = rng.choice(["Get the", "Calculate the", "Store the", "Set the", "Create the", "Compute"])
    tl.insert(i, [f"{ind}# {verb} {human(var)}", tag])
    return ("redundant-comment", "low", "Comment restates the assignment on the next line.", "")


def inj_catch_all(tl, func, rng, tag):
    idx = [i for i in _candidates(tl, func, _CALL_STMT) if not _RETURN.match(tl[i][0])]
    if not idx:
        return None
    i = rng.choice(idx)
    orig = tl[i][0]
    ind = _CALL_STMT.match(orig).group(1)
    exc = rng.choice(["except Exception:", "except Exception as e:", "except:"])
    tl[i:i + 1] = [[f"{ind}try:", tag], [f"{ind}    {orig.strip()}", tag], [f"{ind}{exc}", tag],
                   [f"{ind}    pass", tag]]
    return ("over-defensive", "high",
            f"Catch-all `{exc.rstrip(':')}: pass` silently hides any failure of `{orig.strip()[:60]}`. "
            "Catch the specific exception and handle or log it, or let it raise.", orig)


def inj_none_wrap(tl, func, rng, tag):
    idx = [i for i in _candidates(tl, func, _RETURN_NAME) if _RETURN_NAME.match(tl[i][0]).group(2) not in _KEYWORDS]
    if not idx:
        return None
    i = rng.choice(idx)
    orig = tl[i][0]
    ind, var = _RETURN_NAME.match(orig).groups()
    tl[i:i + 1] = [[f"{ind}if {var} is not None:", tag], [f"{ind}    return {var}", tag],
                   [f"{ind}else:", tag], [f"{ind}    return None", tag]]
    return ("over-defensive", "medium",
            f"The if/else returns `{var}` either way; `return {var}` has identical behaviour.", orig)


_DOC_OPENERS = ["This function is responsible for {h}.", "This method handles the process of {h}.",
                "A robust and comprehensive helper to {h}.", "This function serves as a utility to {h}."]
_DOC_FILLER = ["It takes the provided inputs and processes them accordingly.",
               "This implementation is designed to be efficient, reliable and maintainable.",
               "It gracefully handles all edge cases to ensure correct behaviour.",
               "The function follows best practices for clean and readable code.",
               "This provides a seamless experience for callers of this API."]


def inj_verbose_docstring(tl, func, rng, tag):
    if func.has_docstring or func.body_idx != 1 or not tl[0][0].rstrip().endswith(":"):
        return None
    ind = re.match(r"^(\s*)", tl[1][0]).group(1)
    h = human(func.name)
    body = [rng.choice(_DOC_OPENERS).format(h=h)] + rng.sample(_DOC_FILLER, k=rng.randint(2, 3))
    doc = [f'{ind}"""'] + [f"{ind}{s}" for s in body] + [f"{ind}", f"{ind}Returns:", f"{ind}    The result.",
                                                        f'{ind}"""']
    tl[1:1] = [[d, tag] for d in doc]
    return ("verbose-docstring", "low", "Long docstring made of filler; it never says what the function returns.",
            f'{ind}"""{h.capitalize()}."""')


def inj_emoji_log(tl, func, rng, tag):
    idx = _candidates(tl, func, _RETURN)
    if not idx:
        return None
    i = rng.choice(idx)
    ind = _RETURN.match(tl[i][0]).group(1)
    msg = rng.choice(["✅ {h} completed successfully! 🎉", "🚀 Starting {h}...", "✨ {h} done!", "🔥 {h} finished"])
    tl.insert(i, [f'{ind}logger.info("{msg.format(h=human(func.name).capitalize())}")', tag])
    return ("log-noise", "low", "Emoji/celebratory log message adds noise without information.", "")


def inj_todo(tl, func, rng, tag):
    if rng.random() < 0.6:  # applies almost everywhere; damp it so it doesn't dominate the mix
        return None
    idx = [i for i, (t, g) in enumerate(tl) if g is None and i >= func.body_idx and i not in func.protected
           and t.strip() and _balanced(t)]
    if not idx:
        return None
    i = rng.choice(idx)
    ind = re.match(r"^(\s*)", tl[i][0]).group(1)
    msg = rng.choice(["# TODO: implement proper error handling", "# TODO: add validation here",
                      "# TODO: handle edge cases", "# TODO: replace with real implementation"])
    tl.insert(i, [f"{ind}{msg}", tag])
    return ("dead-or-placeholder", "medium", "Placeholder TODO in new code; do it now or track it in an issue.", "")


INJECTORS = [inj_redundant_comment, inj_catch_all, inj_none_wrap, inj_verbose_docstring, inj_emoji_log, inj_todo]


def to_hunk(func: Func, texts: list[str]) -> Hunk:
    return Hunk(func.path, [(func.start + i, "+", t) for i, t in enumerate(texts)])


def clean_example(func: Func) -> tuple[Hunk, list[Finding]] | None:
    hunk = to_hunk(func, func.lines)
    return None if check_hunk(hunk) else (hunk, [])  # drop "clean" code that already trips a rule


def slop_example(func: Func, rng: random.Random, max_injections: int = 2) -> tuple[Hunk, list[Finding]] | None:
    if clean_example(func) is None:
        return None
    tl = [[t, None] for t in func.lines]
    specs = {}
    want = rng.randint(1, max_injections)
    for inj in rng.sample(INJECTORS, k=len(INJECTORS)):
        if len(specs) >= want:
            break
        tag = len(specs)
        spec = inj(tl, func, rng, tag)
        if spec:
            specs[tag] = spec
    if not specs:
        return None
    hunk = to_hunk(func, [t for t, _ in tl])
    findings = []
    for tag, (cat, sev, reason, fix) in specs.items():
        rows = [i for i, (_, g) in enumerate(tl) if g == tag]
        findings.append(Finding(func.path, func.start + rows[0], cat, sev, reason,
                                end_line=func.start + rows[-1], fix=fix, source="synth"))
    findings.sort(key=lambda f: f.line)
    return hunk, findings
