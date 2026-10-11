"""Post findings to a pull request: one review (inline), one sticky summary, annotations fallback.

Stdlib only (urllib) so the Action installs fast.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter

from .schema import Finding, severity_at_least

API = os.environ.get("GITHUB_API_URL", "https://api.github.com")
SUMMARY_MARKER = "<!-- tidypr:summary -->"
_FP = re.compile(r"<!-- tidypr:fp=([0-9a-f]+) -->")


class GitHubError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"GitHub API {status}: {body[:300]}")
        self.status = status


class GitHub:
    def __init__(self, repo: str, token: str):
        self.repo, self.token = repo, token

    def request(self, method: str, path: str, body: dict | None = None, accept: str = "application/vnd.github+json"):
        url = path if path.startswith("http") else f"{API}/repos/{self.repo}{path}"
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", accept)
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode()
        except urllib.error.HTTPError as e:
            raise GitHubError(e.code, e.read().decode(errors="replace")) from None
        if accept.endswith(".diff"):
            return raw
        return json.loads(raw) if raw else None

    def pr_diff(self, number: int) -> str:
        return self.request("GET", f"/pulls/{number}", accept="application/vnd.github.v3.diff")

    def paginate(self, path: str, pages: int = 5) -> list:
        out: list = []
        for page in range(1, pages + 1):
            batch = self.request("GET", f"{path}?per_page=100&page={page}") or []
            out += batch
            if len(batch) < 100:
                break
        return out


def fingerprint(f: Finding, code: str = "") -> str:
    """Stable across pushes: path + category + normalised code (not line numbers, which shift)."""
    norm = re.sub(r"\s+", " ", code).strip()
    return hashlib.sha1(f"{f.path}|{f.category}|{norm or f.line}".encode()).hexdigest()[:12]


def comment_body(f: Finding, fp: str) -> str:
    body = f"**{f.severity} · {f.category}**\n\n{f.reason}"
    if f.fix is not None:
        body += f"\n\n```suggestion\n{f.fix}\n```"
    return body + f"\n\n<sub>tidypr · {f.source}</sub>\n<!-- tidypr:fp={fp} -->"


def review_comments(findings: list[Finding], code_of: dict[tuple[str, int], str],
                    seen: set[str], min_inline: str) -> list[dict]:
    out = []
    for f in findings:
        if not severity_at_least(f.severity, min_inline):
            continue
        code = "\n".join(code_of.get((f.path, n), "") for n in range(f.span[0], f.span[1] + 1))
        fp = fingerprint(f, code)
        if fp in seen:
            continue
        c = {"path": f.path, "line": f.span[1], "side": "RIGHT", "body": comment_body(f, fp)}
        if f.span[1] > f.span[0]:
            c.update(start_line=f.span[0], start_side="RIGHT")
        out.append(c)
    return out


def summary_body(findings: list[Finding], added_lines: int, model: str | None) -> str:
    if not findings:
        head = "**tidypr: no findings.** ✔"
    else:
        sev = Counter(f.severity for f in findings)
        parts = ", ".join(f"{sev[s]} {s}" for s in ("high", "medium", "low") if sev[s])
        head = f"**tidypr: {len(findings)} finding(s)** ({parts}) on {added_lines} added lines."
    rows = "\n".join(f"| {f.severity} | {f.category} | `{f.path}:{f.line}` | {f.reason.splitlines()[0][:120]} |"
                     for f in findings)
    table = f"\n\n| Severity | Category | Where | Why |\n|---|---|---|---|\n{rows}" if findings else ""
    foot = f"\n\n<sub>model: {model or 'rules only'} · runs inside CI, code never leaves the runner</sub>"
    return f"{SUMMARY_MARKER}\n{head}{table}{foot}"


def upsert_summary(gh: GitHub, number: int, body: str) -> None:
    for c in gh.paginate(f"/issues/{number}/comments"):
        if SUMMARY_MARKER in (c.get("body") or ""):
            gh.request("PATCH", f"/issues/comments/{c['id']}", {"body": body})
            return
    gh.request("POST", f"/issues/{number}/comments", {"body": body})


def existing_fingerprints(gh: GitHub, number: int) -> set[str]:
    return {m for c in gh.paginate(f"/pulls/{number}/comments") for m in _FP.findall(c.get("body") or "")}


def annotate(findings: list[Finding]) -> None:
    """Workflow-command annotations: work without write permission (e.g. fork PRs)."""
    level = {"high": "error", "medium": "warning", "low": "notice"}
    for f in findings:
        msg = f"{f.reason}".replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::{level[f.severity]} file={f.path},line={f.span[0]},endLine={f.span[1]},"
              f"title=tidypr {f.category}::{msg}")


def post(gh: GitHub, number: int, head_sha: str, findings: list[Finding], code_of: dict,
         added_lines: int, mode: str, min_inline: str, model: str | None) -> str:
    """Returns the mode actually used (falls back to annotations on 403/404)."""
    if mode == "annotations":
        annotate(findings)
        return mode
    try:
        upsert_summary(gh, number, summary_body(findings, added_lines, model))
        if mode == "review":
            comments = review_comments(findings, code_of, existing_fingerprints(gh, number), min_inline)
            if comments:
                gh.request("POST", f"/pulls/{number}/reviews", {
                    "commit_id": head_sha, "event": "COMMENT",
                    "body": f"tidypr left {len(comments)} inline comment(s). See the summary comment.",
                    "comments": comments,
                })
        return mode
    except GitHubError as e:
        if e.status in (403, 404, 422):  # read-only token (fork PR), missing perms, bad line
            print(f"::warning::tidypr could not post ({e.status}); falling back to annotations")
            annotate(findings)
            return "annotations"
        raise
