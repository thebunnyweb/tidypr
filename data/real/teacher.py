"""Teacher labelling: a bigger free model turns (hunk + human review comment) into our JSON label.

The human reviewer already found the problem; the teacher only decides whether it is one of our
slop categories and, if so, which lines / severity, and rewrites the point as a short neutral reason.

  python data/real/teacher.py --candidates data/real/candidates.jsonl --out data/real/labeled.jsonl
  (on Kaggle GPU; see data/real/teacher_kaggle.ipynb)

Output line: candidate fields + {"label": "slop"|"not_slop"|"invalid", "finding": {...} | null, "raw": "..."}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tidypr.diff import Hunk  # noqa: E402
from tidypr.schema import CATEGORIES, SEVERITIES  # noqa: E402

TEACHER_SYSTEM = """You label code-review data. You get a diff hunk (numbered lines; '+' = added, ' ' = context) \
and a comment a human reviewer left on it. Decide if the reviewer is pointing out AI-style SLOP of one of these kinds:

""" + "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items()) + """

NOT slop (answer slop=false): bugs and logic errors, naming, formatting/style, performance, security,
missing tests, questions, design debates, requests for MORE code/checks/comments, praise, unclear comments.

If slop=true:
- line/end_line: the added ('+') lines the comment is about (prefer the commented line when given)
- severity: high = hides errors or changes behaviour; medium = real maintenance cost; low = noise
- reason: one short neutral sentence about the code (do NOT mention the reviewer)
- fix: replacement code for line..end_line if obvious, "" to delete those lines, or null if unsure

Reply with JSON only:
{"slop": true, "category": "...", "severity": "...", "line": N, "end_line": N, "reason": "...", "fix": ... }
or {"slop": false}"""

FEWSHOT = [
    ("File: app/sync.py\n```diff\n   40       data = load()\n   41 +     try:\n   42 +         push(data)\n"
     "   43 +     except Exception:\n   44 +         pass\n```\nReviewer comment (on line 44): "
     "\"This silently swallows every error from push, we'll never know it failed\"",
     '{"slop": true, "category": "over-defensive", "severity": "high", "line": 41, "end_line": 44, '
     '"reason": "Catch-all except hides every failure of push().", "fix": "    push(data)"}'),
    ("File: src/api/user.ts\nLanguage: TypeScript\n```diff\n   12 +   const name = user.firstName + user.lastName;\n```\n"
     "Reviewer comment (on line 12): \"Should this have a space between first and last name?\"",
     '{"slop": false}'),
    ("File: lib/report.go\nLanguage: Go\n```diff\n   88 +\t// increment the counter\n   89 +\tcount++\n```\n"
     "Reviewer comment (on line 88): \"nit: this comment doesn't add anything\"",
     '{"slop": true, "category": "redundant-comment", "severity": "low", "line": 88, "end_line": 88, '
     '"reason": "Comment restates the next line.", "fix": ""}'),
]


def to_hunk(row: dict) -> Hunk:
    return Hunk(row["path"], [tuple(x) for x in row["hunk"]])


def user_prompt(row: dict) -> str:
    h = to_hunk(row)
    head = f"File: {row['path']}" + (f"\nLanguage: {row['lang']}" if row.get("lang") else "")
    where = f" (on line {row['anchor']})" if row.get("anchor") else ""
    comment = row["comment"].replace("\r", "").strip()[:1000]
    return f"{head}\n```diff\n{h.render()}\n```\nReviewer comment{where}: \"{comment}\""


def teacher_messages(row: dict) -> list[dict]:
    msgs = [{"role": "system", "content": TEACHER_SYSTEM}]
    for u, a in FEWSHOT:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": user_prompt(row)})
    return msgs


_OBJ = re.compile(r"\{.*\}", re.S)
_MENTIONS_REVIEWER = re.compile(r"\b(the )?(reviewer|commenter)\b|\bas (mentioned|noted)\b", re.I)


def parse_teacher(text: str, row: dict) -> tuple[str, dict | None]:
    """Validate the teacher's answer against the hunk. Returns (label, finding)."""
    m = _OBJ.search(text or "")
    if not m:
        return "invalid", None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "invalid", None
    if d.get("slop") is False:
        return "not_slop", None
    if d.get("slop") is not True:
        return "invalid", None
    try:
        line, end = int(d["line"]), int(d.get("end_line") or d["line"])
    except (KeyError, TypeError, ValueError):
        return "invalid", None
    cat, sev, reason = d.get("category"), d.get("severity"), str(d.get("reason") or "").strip()
    added = {n for n, k, _ in row["hunk"] if k == "+"}
    if cat not in CATEGORIES or sev not in SEVERITIES or not reason or len(reason) > 300 or end < line:
        return "invalid", None
    if not {line, end} <= added or end - line > 25:
        return "invalid", None
    anchor = row.get("anchor")
    if anchor and anchor in added and not (line - 3 <= anchor <= end + 3):
        return "invalid", None           # teacher pointed somewhere other than where the human commented
    if _MENTIONS_REVIEWER.search(reason):
        return "invalid", None
    fix = d.get("fix")
    return "slop", {"line": line, "end_line": end, "category": cat, "severity": sev, "reason": reason,
                    "fix": fix if isinstance(fix, str) else None}


# ---- runners ------------------------------------------------------------------------------------------

def run_vllm(rows, model: str, max_model_len: int, batch: int, quantization: str | None):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype="half", max_model_len=max_model_len, gpu_memory_utilization=0.90,
              quantization=quantization, enforce_eager=False)
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=220, stop=["\n\n\n"])
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        prompts = [tok.apply_chat_template(teacher_messages(r), tokenize=False, add_generation_prompt=True)
                   for r in chunk]
        outs = llm.generate(prompts, params, use_tqdm=False)
        for r, o in zip(chunk, outs):
            yield r, o.outputs[0].text


def run_hf(rows, model: str, batch: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(
        model, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16))
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        prompts = [tok.apply_chat_template(teacher_messages(r), tokenize=False, add_generation_prompt=True)
                   for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=3500).to(m.device)
        with torch.no_grad():
            gen = m.generate(**enc, max_new_tokens=220, do_sample=False, pad_token_id=tok.eos_token_id)
        for r, g in zip(chunk, gen):
            yield r, tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/real/candidates.jsonl")
    ap.add_argument("--out", default="data/real/labeled.jsonl")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
    ap.add_argument("--quantization", default="awq", help="vllm only; '' for none")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.candidates, encoding="utf-8")]
    out = Path(a.out)
    done = set()
    if out.exists():  # resumable: skip rows already labelled
        done = {json.loads(l)["id"] for l in out.open(encoding="utf-8")}
    rows = [r for r in rows if r["id"] not in done]
    if a.limit:
        rows = rows[: a.limit]
    print(f"labelling {len(rows)} candidates ({len(done)} already done) with {a.model}")

    gen = run_vllm(rows, a.model, a.max_model_len, a.batch, a.quantization or None) if a.backend == "vllm" \
        else run_hf(rows, a.model, max(1, a.batch // 8))
    counts = {"slop": 0, "not_slop": 0, "invalid": 0}
    with out.open("a", encoding="utf-8") as fh:
        for n, (r, text) in enumerate(gen, 1):
            label, finding = parse_teacher(text, r)
            counts[label] += 1
            fh.write(json.dumps({**r, "label": label, "finding": finding, "raw": text[:600]}, ensure_ascii=False) + "\n")
            if n % 200 == 0:
                fh.flush()
                print(f"  {n}/{len(rows)}  {counts}")
    print("done:", counts)


if __name__ == "__main__":
    main()
