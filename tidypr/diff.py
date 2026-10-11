"""Minimal unified-diff parser: yields hunks with new-file line numbers for added lines.

Works on `git diff` and on `git log -p` (commit headers/messages between hunks are skipped because
each hunk is read for exactly the line counts its `@@` header declares).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


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
    new_no = old_left = new_left = 0
    for raw in diff_text.splitlines():
        if cur is not None and (old_left > 0 or new_left > 0):
            if raw.startswith("+"):
                cur.lines.append((new_no, "+", raw[1:]))
                new_no += 1
                new_left -= 1
            elif raw.startswith("-"):
                old_left -= 1
            elif raw.startswith(" ") or raw == "":
                cur.lines.append((new_no, " ", raw[1:]))
                new_no += 1
                new_left -= 1
                old_left -= 1
            elif raw.startswith("\\"):   # "\ No newline at end of file"
                pass
            else:                        # malformed/truncated hunk: stop reading it
                cur, old_left, new_left = None, 0, 0
            continue
        if raw.startswith("diff --git"):
            path, cur = None, None
        elif raw.startswith("+++ "):
            target = raw[4:].strip()
            path = None if target == "/dev/null" else target.removeprefix("b/")
        elif (m := _HUNK.match(raw)) and path:
            old_left = int(m.group(1)) if m.group(1) is not None else 1
            new_no = int(m.group(2))
            new_left = int(m.group(3)) if m.group(3) is not None else 1
            cur = Hunk(path)
            hunks.append(cur)
    return [h for h in hunks if h.added_lines]


def code_hunks(hunks: list[Hunk], skip_tests: bool = False, languages: set[str] | None = None) -> list[Hunk]:
    """Hunks in source files of a known language (see tidypr.langs)."""
    from .langs import language_of

    out = [h for h in hunks if language_of(h.path) and (languages is None or language_of(h.path) in languages)]
    if skip_tests:
        out = [h for h in out if not is_test(h.path)]
    return out


def python_hunks(hunks: list[Hunk], skip_tests: bool = False) -> list[Hunk]:
    return code_hunks(hunks, skip_tests, {"Python"})


_TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs", "testing"}
_TEST_SUFFIXES = ("_test.py", "_test.go", ".test.js", ".test.ts", ".test.tsx", ".test.jsx", ".spec.js", ".spec.ts",
                  ".spec.tsx", "_spec.rb", "Test.java", "Tests.java", "Test.kt", "Tests.cs")


def is_test(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    name = parts[-1]
    return bool(_TEST_DIRS & set(parts[:-1])) or name.startswith("test_") or name.endswith(_TEST_SUFFIXES) \
        or name == "conftest.py"


_is_test = is_test
