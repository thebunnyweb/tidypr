"""Minimal unified-diff parser: yields hunks with new-file line numbers for added lines."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass
class Hunk:
    path: str
    # (new_line_number, kind, text) where kind is "+" (added) or " " (context)
    lines: list[tuple[int, str, str]] = field(default_factory=list)

    @property
    def added_lines(self) -> set[int]:
        return {n for n, k, _ in self.lines if k == "+"}

    def render(self) -> str:
        """Numbered view the model sees: `  42 + code` / `  43   code`."""
        return "\n".join(f"{n:>5} {k} {t}" for n, k, t in self.lines)

    def split(self, max_lines: int) -> list["Hunk"]:
        if len(self.lines) <= max_lines:
            return [self]
        return [Hunk(self.path, self.lines[i:i + max_lines]) for i in range(0, len(self.lines), max_lines)]


def parse_diff(diff_text: str) -> list[Hunk]:
    hunks: list[Hunk] = []
    path: str | None = None
    cur: Hunk | None = None
    new_no = 0
    for raw in diff_text.splitlines():
        if raw.startswith("diff --git"):
            path, cur = None, None
        elif raw.startswith("+++ "):
            target = raw[4:].strip()
            path = None if target == "/dev/null" else target.removeprefix("b/")
        elif raw.startswith("--- "):
            continue
        elif (m := _HUNK.match(raw)) and path:
            new_no = int(m.group(1))
            cur = Hunk(path)
            hunks.append(cur)
        elif cur is not None:
            if raw.startswith("+"):
                cur.lines.append((new_no, "+", raw[1:]))
                new_no += 1
            elif raw.startswith(" ") or raw == "":
                cur.lines.append((new_no, " ", raw[1:]))
                new_no += 1
            # "-" lines and "\ No newline" don't exist in the new file
    return [h for h in hunks if h.added_lines]


def python_hunks(hunks: list[Hunk]) -> list[Hunk]:
    return [h for h in hunks if h.path.endswith(".py")]
