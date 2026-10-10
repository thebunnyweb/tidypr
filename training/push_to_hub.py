"""Upload a GGUF + model card to Hugging Face. Used by both training paths.

  export HF_TOKEN=hf_...            # write token from huggingface.co/settings/tokens
  python training/push_to_hub.py --repo you/slop-check-1.5b-gguf --gguf runs/mlx/slop-check-Q4_K_M.gguf
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

CARD = """---
license: apache-2.0
base_model: {base}
tags: [gguf, code-review, python, lora]
---
# {name}

Fine-tune of `{base}` that flags AI slop (over-defensive code, redundant comments, filler
docstrings, emoji logs, placeholder TODOs…) in Python pull-request hunks and returns JSON findings.

Built with [slop-check](https://github.com/thebunnyweb/slop-check). Use it in a PR:

```yaml
- uses: thebunnyweb/slop-check@v0
  with:
    model: {repo}
```

Locally:

```bash
pip install "slop-check[infer]"
git diff main... | slopcheck review --diff - --backend llamacpp --model {repo}
```

**Status:** early. Trained mostly on synthetic injections; see the repo's eval report before trusting it.
"""


def push(repo: str, gguf: Path, base: str, token: str | None = None, private: bool = False) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=str(gguf), path_in_repo=gguf.name, repo_id=repo)
    card = CARD.format(base=base, name=repo.split("/")[-1], repo=repo)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo)
    url = f"https://huggingface.co/{repo}"
    print(f"pushed {gguf.name} → {url}")
    return url


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    push(a.repo, a.gguf, a.base, os.environ.get("HF_TOKEN"), a.private)


if __name__ == "__main__":
    main()
