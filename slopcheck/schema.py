"""Finding schema shared by rules, model output, training data and the GitHub poster."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

CATEGORIES: dict[str, str] = {
    "over-defensive": "catch-all excepts that swallow errors, None/type checks that can't fail, "
                      "if/else wrappers around trivial returns",
    "redundant-comment": "a comment that restates the next line(s) of code",
    "verbose-docstring": "docstring far longer than the function or full of filler "
                         "('robust', 'comprehensive', 'This function is responsible for')",
    "needless-abstraction": "class/factory wrapping one function, unused params or config, "
                            "single-use helpers that only rename",
    "dead-or-placeholder": "TODO stubs, placeholder implementations, commented-out code",
    "log-noise": "emoji or celebratory log messages, print-debugging in non-CLI code",
    "hallucinated-api": "calls to methods or arguments that do not exist in the library",
}

SEVERITIES = ("low", "medium", "high")
_SEV_RANK = {s: i for i, s in enumerate(SEVERITIES)}


def severity_at_least(sev: str, threshold: str) -> bool:
    return _SEV_RANK.get(sev, -1) >= _SEV_RANK[threshold]


@dataclass
class Finding:
    path: str
    line: int
    category: str
    severity: str
    reason: str
    end_line: int | None = None
    fix: str | None = None          # replacement code for line..end_line, or None for prose-only
    source: str = "model"           # "model" | "rules"
    extra: dict = field(default_factory=dict)

    @property
    def span(self) -> tuple[int, int]:
        return self.line, self.end_line or self.line

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("extra")
        return d

    def to_target(self) -> dict:
        """The shape the model is trained to emit (no path/source)."""
        d = {"line": self.line, "end_line": self.end_line or self.line, "category": self.category,
             "severity": self.severity, "reason": self.reason}
        if self.fix is not None:
            d["fix"] = self.fix
        return d


# JSON schema for constrained decoding (llama.cpp) and validation.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "severity": {"type": "string", "enum": list(SEVERITIES)},
                    "reason": {"type": "string"},
                    "fix": {"type": "string"},
                },
                "required": ["line", "category", "severity", "reason"],
            },
        }
    },
    "required": ["findings"],
}

_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def parse_model_output(text: str, path: str, valid_lines: set[int] | None = None) -> list[Finding]:
    """Parse model text into findings, dropping anything malformed or outside the hunk."""
    m = _JSON_OBJ.search(text or "")
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out: list[Finding] = []
    for item in data.get("findings", []) if isinstance(data, dict) else []:
        try:
            line = int(item["line"])
            end = int(item.get("end_line") or line)
            cat, sev = item["category"], item["severity"]
            reason = str(item["reason"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if cat not in CATEGORIES or sev not in SEVERITIES or not reason or end < line:
            continue
        if valid_lines is not None and not {line, end} <= valid_lines:
            continue
        fix = item.get("fix")
        out.append(Finding(path=path, line=line, end_line=end, category=cat, severity=sev,
                           reason=reason, fix=fix if isinstance(fix, str) else None))
    return out
