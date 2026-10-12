import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "data" / "real"))

from collect import _last_kind, hunk_from_diff_hunk, slop_score  # noqa: E402
from teacher import parse_teacher, teacher_messages, user_prompt  # noqa: E402

from tidypr.diff import code_hunks, is_test, parse_diff
from tidypr.prompt import user_message

DH = ("@@ -10,4 +10,7 @@ def send(self):\n     data = self.load()\n+    try:\n+        self.client.post(data)\n"
      "+    except Exception:\n+        pass")


def row(**kw):
    h = hunk_from_diff_hunk("app/x.py", DH)
    r = {"id": "1", "source": "aidev", "repo": "o/r", "path": "app/x.py", "lang": "Python",
         "hunk": [list(x) for x in h.lines], "anchor": h.lines[-1][0], "comment": "swallows errors"}
    r.update(kw)
    return r


def test_github_diff_hunk_anchor_is_last_line():
    h = hunk_from_diff_hunk("app/x.py", DH)
    assert h.lines[-1] == (14, "+", "        pass") and _last_kind(DH) == "+"


def test_slop_score_ranks_slop_comments_above_others():
    assert slop_score("this silently swallows the error, unnecessary try") > slop_score("rename to snake_case") == 0


def test_teacher_prompt_includes_hunk_comment_and_anchor():
    p = user_prompt(row())
    assert "13 +     except Exception:" in p and "(on line 14)" in p and "swallows errors" in p
    assert teacher_messages(row())[-1]["content"] == p


def test_parse_teacher_accepts_valid_and_rejects_bad_spans():
    ok = '{"slop": true, "category": "over-defensive", "severity": "high", "line": 11, "end_line": 14, ' \
         '"reason": "Catch-all hides failures of post().", "fix": "    self.client.post(data)"}'
    label, f = parse_teacher("sure! " + ok, row())
    assert label == "slop" and f["line"] == 11 and f["fix"].strip() == "self.client.post(data)"
    assert parse_teacher('{"slop": false}', row()) == ("not_slop", None)
    label, f = parse_teacher(ok.replace('"line": 11', '"line": 10'), row())    # span starts on a context line
    assert label == "slop" and (f["line"], f["end_line"]) == (11, 14)          # → clipped to the added lines
    assert parse_teacher(ok.replace("over-defensive", "made-up"), row())[0] == "invalid"
    assert parse_teacher(ok.replace("Catch-all", "The reviewer says catch-all"), row())[0] == "invalid"
    assert parse_teacher("no json here", row())[0] == "invalid"
    far = row(hunk=[[n, "+", "x"] for n in range(1, 40)], anchor=35)
    assert parse_teacher(ok, far)[0] == "invalid"   # snapped to line 35 (where the human commented): no try/except there


def test_language_aware_hunks_and_tests():
    diff = ("diff --git a/src/a.ts b/src/a.ts\n--- a/src/a.ts\n+++ b/src/a.ts\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/src/a.test.ts b/src/a.test.ts\n--- a/src/a.test.ts\n+++ b/src/a.test.ts\n@@ -1 +1 @@\n-a\n+b\n")
    hunks = code_hunks(parse_diff(diff), skip_tests=True)
    assert [h.path for h in hunks] == ["src/a.ts"]
    assert "Language: TypeScript" in user_message(hunks[0])
    assert is_test("pkg/foo_test.go") and is_test("src/__tests__/x.js") and not is_test("src/testing_utils_x.py")


def test_consistency_checks_reject_labels_that_dont_match_the_code():
    from teacher import consistent
    assert consistent("redundant-comment", "Comment restates the code.", ["    // Listen for changes"])
    assert not consistent("dead-or-placeholder", "TODO stub.", ["    // Listen for changes"])
    assert consistent("dead-or-placeholder", "TODO left behind.", ["  # TODO: implement"])
    assert consistent("log-noise", "Leftover debug log.", ["      console.log(values);"])
    assert not consistent("log-noise", "Emoji log.", ["    total += 1"])
    assert not consistent("redundant-comment", "Comment restates.", ["    return x"])


def test_merge_attaches_sibling_findings_to_overlapping_hunks(tmp_path):
    from collect import _row
    from merge import load_real_slop
    base = "@@ -1,2 +1,6 @@\n func a() {\n+\t// Call the handler\n+\th()\n+\t// Delegate\n+\treturn d()"
    h1 = hunk_from_diff_hunk("x.go", base.rsplit("\n", 2)[0])      # ends at line 3 (first comment)
    h2 = hunk_from_diff_hunk("x.go", base)                          # ends at line 5 (second comment)
    def lab(h, line):
        r = _row("aidev", "o/r", "x.go", "Go", h, line, "remove this comment", pr="pr1")
        r.update(label="slop", finding={"line": line, "end_line": line, "category": "redundant-comment",
                                        "severity": "low", "reason": "Comment restates the code.", "fix": ""})
        return r
    p = tmp_path / "l.jsonl"
    p.write_text("\n".join(json.dumps(lab(h, n)) for h, n in [(h1, 2), (h2, 4)]))
    exs = sorted(load_real_slop([str(p)], max_cat_share=1.0), key=lambda e: len(e["hunk"].lines))
    assert [f.line for f in exs[0]["findings"]] == [2]
    assert [f.line for f in exs[1]["findings"]] == [2, 4]     # sibling's line 2 is labelled here too


def test_parse_teacher_repairs_and_explains():
    from teacher import parse_teacher_explain
    r = row(comment="why is this needed?")
    vague = '{"slop": true, "category": "needless-abstraction", "line": 11, "end_line": 14, "reason": "Not needed."}'
    assert parse_teacher_explain(vague, r)[:1] == ("not_slop",) and parse_teacher_explain(vague, r)[2] == "vague-comment"
    alias = '{"slop": true, "category": "dead-code", "line": 14, "end_line": 14, "reason": "Leftover stub."}'
    label, f, _ = parse_teacher_explain(alias, row())
    assert label == "slop" and f["category"] == "dead-or-placeholder" and f["severity"] == "medium"
    assert parse_teacher_explain("nope", row())[2] == "no-json"


def test_reconcile_fixes_category_from_reason():
    from tidypr.checks import reconcile
    assert reconcile("needless-abstraction", "Import is unused.") == "dead-or-placeholder"
    assert reconcile("needless-abstraction", "Unused variable and constant.") == "dead-or-placeholder"
    assert reconcile("redundant-comment", "Second condition is redundant due to the first.") == "over-defensive"
    assert reconcile("redundant-comment", "Comment restates the next line.") == "redundant-comment"
    assert reconcile("redundant-comment", "Comment mentions an unused flag.") == "redundant-comment"
    assert reconcile("verbose-docstring", "Docstring is too short and lacks detail.") is None
    assert reconcile("verbose-docstring", "Docstring is overly verbose.") == "verbose-docstring"
    assert reconcile("dead-or-placeholder", "Change appears to be a leftover from another commit.") is None
    assert reconcile("log-noise", "Unused debug print.") == "log-noise"
    assert reconcile("over-defensive", "Catch-all swallows errors.") == "over-defensive"
