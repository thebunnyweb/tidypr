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
    assert parse_teacher(ok.replace('"line": 11', '"line": 10'), row())[0] == "invalid"       # context line
    assert parse_teacher(ok.replace("over-defensive", "made-up"), row())[0] == "invalid"
    assert parse_teacher(ok.replace("Catch-all", "The reviewer says catch-all"), row())[0] == "invalid"
    assert parse_teacher("no json here", row())[0] == "invalid"
    far = row(hunk=[[n, "+", "x"] for n in range(1, 40)], anchor=35)
    assert parse_teacher(ok, far)[0] == "invalid"   # points far away from where the human commented


def test_language_aware_hunks_and_tests():
    diff = ("diff --git a/src/a.ts b/src/a.ts\n--- a/src/a.ts\n+++ b/src/a.ts\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/src/a.test.ts b/src/a.test.ts\n--- a/src/a.test.ts\n+++ b/src/a.test.ts\n@@ -1 +1 @@\n-a\n+b\n")
    hunks = code_hunks(parse_diff(diff), skip_tests=True)
    assert [h.path for h in hunks] == ["src/a.ts"]
    assert "Language: TypeScript" in user_message(hunks[0])
    assert is_test("pkg/foo_test.go") and is_test("src/__tests__/x.js") and not is_test("src/testing_utils_x.py")
