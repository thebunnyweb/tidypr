"""File extension → language. The model reviews any of these; the regex rules only know Python."""
from __future__ import annotations

from pathlib import PurePosixPath

LANGS: dict[str, str] = {
    ".py": "Python", ".pyi": "Python",
    ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".go": "Go", ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin", ".scala": "Scala",
    ".rb": "Ruby", ".php": "PHP", ".rs": "Rust", ".swift": "Swift", ".dart": "Dart",
    ".c": "C", ".h": "C", ".cc": "C++", ".cpp": "C++", ".cxx": "C++", ".hpp": "C++", ".cs": "C#",
    ".sh": "Shell", ".bash": "Shell", ".lua": "Lua", ".ex": "Elixir", ".exs": "Elixir",
    ".vue": "Vue", ".svelte": "Svelte",
}
# CodeReviewer and other datasets name languages their own way
ALIASES = {"py": "Python", "python": "Python", "js": "JavaScript", "javascript": "JavaScript",
           "ts": "TypeScript", "typescript": "TypeScript", "go": "Go", "golang": "Go", "java": "Java",
           "rb": "Ruby", "ruby": "Ruby", "php": "PHP", "c": "C", "cpp": "C++", "c++": "C++",
           ".cs": "C#", "cs": "C#", "c#": "C#", "csharp": "C#", "rust": "Rust", "kotlin": "Kotlin",
           "swift": "Swift", "scala": "Scala"}


def language_of(path: str) -> str | None:
    return LANGS.get(PurePosixPath(path).suffix.lower())


def normalise(lang: str | None) -> str | None:
    if not lang:
        return None
    return ALIASES.get(str(lang).strip().lower(), str(lang).strip())
