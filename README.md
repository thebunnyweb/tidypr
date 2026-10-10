# slop-check

A small fine-tuned model that flags **AI slop** in Python pull requests (catch-all excepts that swallow errors, comments that restate code, filler docstrings, emoji logs, placeholder TODOs, needless if/else) and posts focused comments on the PR. It runs inside CI on CPU, so it costs nothing per PR and code never leaves the runner.

Status: **v0.1, rough start.** The rule pass works today. Train a model with one of the two paths below to add the parts rules can't catch.

```
data/build_dataset.py  ──►  train.jsonl / valid.jsonl  (chat format, shared by both paths)
                               │
           ┌───────────────────┴───────────────────┐
  training/train_unsloth.py              training/train_mlx.py
  NVIDIA: Kaggle / Colab / RunPod        Apple Silicon: your Mac
  → GGUF directly                        → fuse → training/export_gguf.py → GGUF
           └───────────────────┬───────────────────┘
                 training/push_to_hub.py  →  huggingface.co/<you>/slop-check-*-gguf
                               │
             action.yml (GitHub Action)  →  llama.cpp on the runner  →  PR comments
```

## 0. Install and try it (rules only, no model)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
slopcheck review --diff data/seed/sample_pr.diff
git diff main... | slopcheck review --diff -
pytest -q
```

## 1. Build training data (free)

```bash
pip install -e ".[data]"
# Offline: CPython's stdlib as the source of clean human code
python data/build_dataset.py --out data/out --max-examples 4000 \
  --source-dir "$(python -c 'import sysconfig;print(sysconfig.get_paths()["stdlib"])')"
# More data: add --commitpackft 30000 (streams bigcode/commitpackft from Hugging Face)
```

This takes clean functions and injects *known* slop, so labels and fixes are exact. About 45% of examples are left clean (`{"findings": []}`) to teach the model to stay quiet.

## 2a. Train on a GPU with Unsloth (Kaggle, free)

Open `training/kaggle_unsloth.ipynb` on Kaggle (GPU T4, Internet on, secret `HF_TOKEN`), set `HF_REPO`, and run all cells. Or on any CUDA machine:

```bash
pip install -e ".[train-unsloth]"
python training/train_unsloth.py --data data/out --out runs/unsloth --gguf q4_k_m \
  --push you/slop-check-1.5b-gguf          # HF_TOKEN must be set
python training/train_unsloth.py --max-steps 30   # 5-minute smoke test
```

## 2b. Train locally on a Mac with MLX

```bash
pip install -e ".[train-mlx]"
python training/train_mlx.py --data data/out --out runs/mlx --fuse     # 0.5B default; fits 8–16 GB
# 16 GB+: --base Qwen/Qwen2.5-Coder-1.5B-Instruct. Short on memory: --batch 1 --num-layers 8 --max-seq 2048

# try the adapter before exporting
slopcheck review --diff data/seed/sample_pr.diff --backend mlx \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct --adapter runs/mlx/adapters

# GGUF for CI (needs a llama.cpp checkout + `brew install llama.cpp`; see the script's docstring)
python training/export_gguf.py --hf-dir runs/mlx/fused --out runs/mlx/slop-check-0.5b-Q4_K_M.gguf
```

## 3. Host on Hugging Face

```bash
export HF_TOKEN=hf_...      # write token: huggingface.co/settings/tokens
python training/push_to_hub.py --repo you/slop-check-0.5b-gguf --gguf runs/mlx/slop-check-0.5b-Q4_K_M.gguf
```

The Unsloth path does this itself when you pass `--push`. Both paths produce the same repo layout: one `.gguf` and a model card.

Test the hosted model from anywhere:

```bash
pip install -e ".[infer]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
slopcheck review --diff data/seed/sample_pr.diff --backend llamacpp --model you/slop-check-0.5b-gguf
```

## 4. Use it on your PRs

Add `.github/workflows/slop-check.yml` to any repo:

```yaml
name: slop-check
on: { pull_request: { types: [opened, synchronize, reopened] } }
permissions: { contents: read, pull-requests: write }
jobs:
  slop:
    runs-on: ubuntu-latest
    steps:
      - uses: thebunnyweb/slop-check@main
        with:
          model: you/slop-check-0.5b-gguf   # empty = rules only
          post_mode: summary-only           # → `review` for inline suggestions once you trust it
          fail_on: ""                       # e.g. `high` to block merges
```

| Input | Default | Meaning |
|---|---|---|
| `model` | `""` | HF repo with the GGUF (cached between runs) |
| `model_file` | `*Q4_K_M.gguf` | file/glob inside the repo |
| `post_mode` | `summary-only` | `summary-only`, `review` (inline + suggestion buttons), or `annotations` |
| `min_severity` | `low` | drop anything below this |
| `min_inline` | `medium` | lowest severity that gets an inline comment |
| `fail_on` | `""` | fail the job at this severity or above |
| `max_findings` | `10` | hard cap per PR |

The summary comment is edited in place on every push. Inline comments carry a fingerprint, so they're never posted twice. PRs from forks fall back to annotations, because their token is read-only.

## 5. Evaluate

```bash
python eval/evaluate.py --data data/out/valid.jsonl                       # rules baseline
python eval/evaluate.py --data data/out/valid.jsonl --backend llamacpp --model you/slop-check-0.5b-gguf --no-rules
```

The validation set is synthetic, so the rules look better on it than they will on real code. The numbers that count come from the hand-labelled gold set (`data/gold/`, next milestone).

## Layout

| Path | What |
|---|---|
| `slopcheck/diff.py` | unified diff → hunks with new-file line numbers |
| `slopcheck/rules.py` | zero-dep rule pass |
| `slopcheck/prompt.py` | the one prompt format used by training **and** inference |
| `slopcheck/schema.py` | categories, `Finding`, JSON schema, output parser |
| `slopcheck/synth.py` | slop injectors for synthetic data |
| `slopcheck/infer.py` | backends (`rules`, `llamacpp`, `mlx`) and the review pipeline |
| `slopcheck/github.py` | summary comment, review with suggestions, annotations fallback |
| `training/` | Unsloth, MLX, GGUF export, HF push, Kaggle notebook |
| `eval/evaluate.py` | per-category P/R/F1 and the false-positive gate |
| `docs/PLAN.md` | the full plan |
