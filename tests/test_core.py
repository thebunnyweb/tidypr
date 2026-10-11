import json
import random
from pathlib import Path

from tidypr.diff import Hunk, parse_diff, python_hunks
from tidypr.github import comment_body, fingerprint, review_comments, summary_body
from tidypr.infer import Backend, ReviewConfig, review_diff, review_hunks
from tidypr.prompt import build_messages
from tidypr.rules import check_hunk
from tidypr.schema import Finding, parse_model_output
from tidypr.synth import extract_functions, slop_example

SAMPLE = (Path(__file__).parents[1] / "data/seed/sample_pr.diff").read_text()


def hunk(*lines, start=1, path="a.py"):
    return Hunk(path, [(start + i, "+", t) for i, t in enumerate(lines)])


def cats(findings):
    return sorted(f.category for f in findings)


# ---- diff -------------------------------------------------------------------

def test_parse_diff_numbers_added_and_context_lines():
    diff = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -10,3 +10,4 @@\n ctx\n-old\n+new1\n+new2\n ctx2\n"
    [h] = parse_diff(diff)
    assert h.path == "x.py"
    assert h.lines == [(10, " ", "ctx"), (11, "+", "new1"), (12, "+", "new2"), (13, " ", "ctx2")]
    assert h.added_lines == {11, 12}


def test_deleted_files_and_non_python_are_skipped():
    diff = ("diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
            "diff --git a/r.md b/r.md\n--- a/r.md\n+++ b/r.md\n@@ -1 +1 @@\n-a\n+b\n")
    assert python_hunks(parse_diff(diff)) == []


# ---- rules ------------------------------------------------------------------

def test_rules_on_sample_pr():
    found = cats(review_diff(SAMPLE, None, ReviewConfig(max_findings=50)))
    assert found.count("over-defensive") == 1
    assert "log-noise" in found and "verbose-docstring" in found
    assert found.count("dead-or-placeholder") == 2       # TODO + commented-out block
    assert found.count("redundant-comment") >= 2


def test_rules_quiet_on_clean_code():
    clean = hunk("def area(r):", '    """Return the area of a circle."""', "    # tau/2 keeps this exact",
                 "    return math.pi * r * r")
    assert check_hunk(clean) == []


def test_specific_except_is_not_flagged():
    assert check_hunk(hunk("try:", "    x()", "except KeyError:", "    pass")) == []


# ---- model output parsing -----------------------------------------------------

def test_parse_model_output_drops_bad_items_and_lines_outside_hunk():
    text = 'noise {"findings": [' \
           '{"line": 2, "category": "log-noise", "severity": "low", "reason": "emoji"},' \
           '{"line": 99, "category": "log-noise", "severity": "low", "reason": "outside"},' \
           '{"line": 2, "category": "made-up", "severity": "low", "reason": "x"}]} trailing'
    out = parse_model_output(text, "a.py", valid_lines={1, 2, 3})
    assert [(f.line, f.category) for f in out] == [(2, "log-noise")]
    assert parse_model_output("not json", "a.py") == []


class FakeBackend(Backend):
    def __init__(self, reply):
        self.reply, self.seen = reply, []

    def generate(self, messages):
        self.seen.append(messages)
        return self.reply


def test_model_and_rules_are_merged_and_deduped():
    h = hunk("try:", "    go()", "except Exception:", "    pass")
    reply = json.dumps({"findings": [{"line": 3, "end_line": 4, "category": "over-defensive", "severity": "high",
                                      "reason": "model says", "fix": "except OSError:\n    log()"}]})
    out = review_hunks([h], FakeBackend(reply))
    assert len(out) == 1 and out[0].source == "model" and out[0].fix


# ---- synthetic data -------------------------------------------------------------

SRC = '''
def load(path, retries):
    data = read_file(path)
    parsed = parse(data)
    validate(parsed)
    result = transform(parsed, retries)
    return result
'''


def test_synth_labels_point_at_injected_lines():
    [func] = extract_functions(SRC, "m.py")
    rng = random.Random(0)
    for _ in range(50):
        ex = slop_example(func, rng)
        if not ex:
            continue
        h, findings = ex
        text = dict((n, t) for n, _, t in h.lines)
        for f in findings:
            assert f.line in text and f.end_line in text
            snippet = "\n".join(text[n] for n in range(f.line, f.end_line + 1))
            marker = {"redundant-comment": "#", "over-defensive": ("except", "if "), "log-noise": "logger",
                      "verbose-docstring": '"""', "dead-or-placeholder": "TODO"}[f.category]
            assert any(m in snippet for m in (marker if isinstance(marker, tuple) else (marker,)))
        msgs = build_messages(h, findings)
        assert json.loads(msgs[2]["content"])["findings"][0]["line"] == findings[0].line


# ---- github payloads --------------------------------------------------------------

def test_review_comment_payload_and_dedupe():
    f = Finding("a.py", 3, "over-defensive", "high", "swallows", end_line=4, fix="except OSError:")
    code = {("a.py", 3): "except Exception:", ("a.py", 4): "    pass"}
    [c] = review_comments([f], code, seen=set(), min_inline="medium")
    assert c["start_line"] == 3 and c["line"] == 4 and "```suggestion\nexcept OSError:\n```" in c["body"]
    fp = fingerprint(f, "except Exception:\n    pass")
    assert review_comments([f], code, seen={fp}, min_inline="medium") == []
    low = Finding("a.py", 1, "log-noise", "low", "emoji")
    assert review_comments([low], code, set(), "medium") == []
    assert fp in comment_body(f, fp)


def test_summary_body():
    body = summary_body([Finding("a.py", 1, "log-noise", "low", "emoji")], 10, None)
    assert body.startswith("<!-- tidypr:summary -->") and "1 low" in body and "rules only" in body


def test_emoji_fix_is_clean():
    [f] = check_hunk(hunk('    logger.info("✅ Report sent successfully! 🎉")'))
    assert f.fix == '    logger.info("Report sent successfully!")'


# ---- posting flow (no network) ----------------------------------------------------

class FakeGH:
    def __init__(self, fail=None, existing_summary=False):
        self.calls, self.fail, self.existing = [], fail, existing_summary

    def request(self, method, path, body=None, accept=None):
        from tidypr.github import GitHubError
        self.calls.append((method, path))
        if self.fail and method == "POST":
            raise GitHubError(self.fail, "nope")
        return None

    def paginate(self, path, pages=5):
        if path.endswith("/comments") and "issues" in path and self.existing:
            return [{"id": 7, "body": "<!-- tidypr:summary --> old"}]
        return []


def test_post_review_mode_updates_sticky_summary_and_posts_review():
    from tidypr.github import post
    gh = FakeGH(existing_summary=True)
    f = Finding("a.py", 3, "over-defensive", "high", "swallows", end_line=4)
    used = post(gh, 1, "sha", [f], {}, 10, mode="review", min_inline="medium", model=None)
    assert used == "review"
    assert ("PATCH", "/issues/comments/7") in gh.calls and ("POST", "/pulls/1/reviews") in gh.calls


def test_post_falls_back_to_annotations_on_fork_pr(capsys):
    from tidypr.github import post
    gh = FakeGH(fail=403)
    f = Finding("a.py", 3, "over-defensive", "high", "swallows", end_line=4)
    assert post(gh, 1, "sha", [f], {}, 10, mode="review", min_inline="medium", model=None) == "annotations"
    assert "::error file=a.py,line=3,endLine=4" in capsys.readouterr().out
