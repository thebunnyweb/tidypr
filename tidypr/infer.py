"""Model backends + the end-to-end review pipeline.

Backends:
  rules     no model, rule pass only (default; zero deps)
  llamacpp  GGUF via llama-cpp-python, from a Hugging Face repo or a local file (CI path)
  mlx       MLX on Apple Silicon, from a HF repo or local dir / adapter (Mac testing path)
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .checks import consistent
from .diff import Hunk, code_hunks, parse_diff
from .prompt import build_messages
from .rules import check_hunk
from .schema import OUTPUT_SCHEMA, Finding, parse_model_output, severity_at_least

MAX_HUNK_LINES = 120
MAX_NEW_TOKENS = 768


class Backend:
    def generate(self, messages: list[dict]) -> str:
        raise NotImplementedError


class LlamaCppBackend(Backend):
    def __init__(self, model: str, filename: str = "*Q4_K_M.gguf", n_ctx: int = 4096, threads: int | None = None):
        from llama_cpp import Llama

        kw = dict(n_ctx=n_ctx, n_threads=threads, verbose=False)
        if model.endswith(".gguf"):
            self.llm = Llama(model_path=model, **kw)
        else:  # Hugging Face repo id; downloads once into ~/.cache/huggingface
            self.llm = Llama.from_pretrained(repo_id=model, filename=filename, **kw)

    def generate(self, messages: list[dict]) -> str:
        out = self.llm.create_chat_completion(
            messages=messages, temperature=0.0, max_tokens=MAX_NEW_TOKENS,
            response_format={"type": "json_object", "schema": OUTPUT_SCHEMA},
        )
        return out["choices"][0]["message"]["content"]


class MLXBackend(Backend):
    def __init__(self, model: str, adapter_path: str | None = None):
        from mlx_lm import load

        self.model, self.tokenizer = load(model, adapter_path=adapter_path)

    def generate(self, messages: list[dict]) -> str:
        from mlx_lm import generate

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=MAX_NEW_TOKENS)


def make_backend(name: str, model: str | None = None, **kw) -> Backend | None:
    if name == "rules":
        return None
    if not model:
        raise ValueError(f"backend '{name}' needs --model (HF repo id or local path)")
    if name == "llamacpp":
        return LlamaCppBackend(model, **{k: v for k, v in kw.items() if k in ("filename", "n_ctx", "threads")})
    if name == "mlx":
        return MLXBackend(model, adapter_path=kw.get("adapter_path"))
    raise ValueError(f"unknown backend: {name}")


# Categories a hunk-only model can judge reliably. needless-abstraction / hallucinated-api need context
# the model doesn't see; enable them explicitly once the test split says they're trustworthy.
DEFAULT_CATEGORIES = frozenset({"over-defensive", "redundant-comment", "verbose-docstring",
                                "dead-or-placeholder", "log-noise"})


@dataclass
class ReviewConfig:
    min_severity: str = "low"
    max_findings: int = 10
    use_rules: bool = True
    categories: frozenset[str] = DEFAULT_CATEGORIES


def review_hunks(hunks: list[Hunk], backend: Backend | None, cfg: ReviewConfig = ReviewConfig()) -> list[Finding]:
    findings: list[Finding] = []
    for big in hunks:
        for hunk in big.split(MAX_HUNK_LINES):
            if cfg.use_rules and hunk.path.endswith(".py"):  # rules are Python-only
                findings += check_hunk(hunk)
            if backend is not None:
                text = backend.generate(build_messages(hunk))
                code = {n: t for n, _, t in hunk.lines}
                for f in parse_model_output(text, hunk.path, valid_lines=hunk.added_lines):
                    if consistent(f.category, f.reason, [code[n] for n in range(f.span[0], f.span[1] + 1) if n in code]):
                        findings.append(f)   # drop model findings that don't match the code
    findings = dedupe(findings)
    findings = [f for f in findings if severity_at_least(f.severity, cfg.min_severity) and f.category in cfg.categories]
    findings.sort(key=lambda f: (-["low", "medium", "high"].index(f.severity), f.path, f.line))
    return findings[: cfg.max_findings]


def review_diff(diff_text: str, backend: Backend | None, cfg: ReviewConfig = ReviewConfig()) -> list[Finding]:
    return review_hunks(code_hunks(parse_diff(diff_text)), backend, cfg)


def dedupe(findings: list[Finding]) -> list[Finding]:
    """Same path + category + overlapping lines → keep one (prefer model: it carries a better fix)."""
    kept: list[Finding] = []
    for f in sorted(findings, key=lambda f: f.source != "model"):
        a0, a1 = f.span
        if any(k.path == f.path and k.category == f.category and k.span[0] <= a1 and a0 <= k.span[1]
               for k in kept):
            continue
        kept.append(f)
    return kept


def findings_json(findings: list[Finding]) -> str:
    return json.dumps([f.to_dict() for f in findings], indent=2, ensure_ascii=False)
