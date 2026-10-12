# tidypr

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
                 training/push_to_hub.py  →  huggingface.co/<you>/tidypr-*-gguf
                               │
             action.yml (GitHub Action)  →  llama.cpp on the runner  →  PR comments
```

## 0. Install and try it (rules only, no model)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
tidypr review --diff data/seed/sample_pr.diff
git diff main... | tidypr review --diff -
pytest -q
```

## 1. Build training data (free)

```bash
pip install -e ".[data]"
STDLIB="$(python -c 'import sysconfig;print(sysconfig.get_paths()["stdlib"])')"
python data/build_dataset.py --out data/v2 --max-examples 6000 \
  --commitpackft 40000=3500 \
  --source ~/code/your-old-repo=1000 \
  --source "$STDLIB=1500"
```

Each `--source DIR=MAX` caps how many examples come from that source. Sources are interleaved, so the mix is what you asked for.

Every example is one of three kinds (`--mix clean,decoy,slop`, default `0.30,0.20,0.50`):
- **clean:** real code, untouched → `{"findings": []}`
- **decoy:** code that *looks* like slop but is fine (a specific `except` that logs, a comment that explains *why*, `if arg is None: arg = []`, plain logs, tracked `TODO(#123)`) → `{"findings": []}`
- **slop:** known slop injected (catch-all excepts, `raise e` re-raises, pointless if/else, comments that narrate, filler docstrings, emoji logs, vague TODOs) → exact labels

Hunks look like real diffs: partial edits with context lines, not only whole new functions.

## 1b. Label a gold set from real PRs (the honest test)

```bash
git -C ~/code/some-repo log -p --since=2025-01-01 -- '*.py' > /tmp/real.diff
python data/gold/label.py /tmp/real.diff        # Enter = clean, or "27-28 od h reason"
python eval/evaluate.py --data data/gold/gold.jsonl                     # rules
python eval/evaluate.py --data data/gold/gold.jsonl --no-rules \
  --backend llamacpp --model bunnyweb/tidypr-1.5b-gguf                  # model
```

Never train on `data/gold/`. Synthetic validation scores look great; only the gold set tells you whether the model works on real code.

## 2a. Train on a GPU with Unsloth (Kaggle, free)

Follow the steps at the top of `training/kaggle_unsloth.ipynb`: zip your dataset folder (e.g. `cd data && zip -r ../tidypr-data.zip v2`), upload it as a private Kaggle Dataset, import the notebook, pick **GPU T4 x1** (not P100), turn Internet on, add the `HF_TOKEN` secret, set `HF_REPO`, Run All. Or on any CUDA machine:

```bash
pip install -e ".[train-unsloth]"
python training/train_unsloth.py --data data/v2 --out runs/unsloth --gguf q8_0 \
  --push bunnyweb/tidypr-1.5b-gguf          # HF_TOKEN must be set
python training/train_unsloth.py --max-steps 30   # 5-minute smoke test
```

## 2b. Train locally on a Mac with MLX

```bash
pip install -e ".[train-mlx]"
python training/train_mlx.py --data data/v2 --out runs/mlx --fuse     # 0.5B default; fits 8–16 GB
# 16 GB+: --base Qwen/Qwen2.5-Coder-1.5B-Instruct. Short on memory: --batch 1 --num-layers 8 --max-seq 2048

# try the adapter before exporting
tidypr review --diff data/seed/sample_pr.diff --backend mlx \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct --adapter runs/mlx/adapters

# GGUF for CI (needs a llama.cpp checkout + `brew install llama.cpp`; see the script's docstring)
python training/export_gguf.py --hf-dir runs/mlx/fused --out runs/mlx/tidypr-0.5b-Q4_K_M.gguf
```

## 3. Host on Hugging Face

```bash
export HF_TOKEN=hf_...      # write token: huggingface.co/settings/tokens
python training/push_to_hub.py --repo bunnyweb/tidypr-0.5b-gguf --gguf runs/mlx/tidypr-0.5b-Q4_K_M.gguf
```

The Unsloth path does this itself when you pass `--push`. Both paths produce the same repo layout: one `.gguf` and a model card.

Test the hosted model from anywhere:

```bash
pip install -e ".[infer]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
tidypr review --diff data/seed/sample_pr.diff --backend llamacpp --model bunnyweb/tidypr-1.5b-gguf
```

## 4. Use it on your PRs

Add `.github/workflows/tidypr.yml` to any repo:

```yaml
name: tidypr
on: { pull_request: { types: [opened, synchronize, reopened] } }
permissions: { contents: read, pull-requests: write }
jobs:
  slop:
    runs-on: ubuntu-latest
    steps:
      - uses: thebunnyweb/tidypr@main
        with:
          model: bunnyweb/tidypr-1.5b-gguf   # empty = rules only
          post_mode: summary-only           # → `review` for inline suggestions once you trust it
          fail_on: ""                       # e.g. `high` to block merges
```

| Input | Default | Meaning |
|---|---|---|
| `model` | `""` | HF repo with the GGUF (cached between runs) |
| `model_file` | `*Q8_0.gguf` | file/glob inside the repo (Q8_0 keeps recall; Q4_K_M lost about 40% of findings in testing) |
| `post_mode` | `summary-only` | `summary-only`, `review` (inline + suggestion buttons), or `annotations` |
| `min_severity` | `low` | drop anything below this |
| `min_inline` | `medium` | lowest severity that gets an inline comment |
| `fail_on` | `""` | fail the job at this severity or above |
| `max_findings` | `10` | hard cap per PR |

The summary comment is edited in place on every push. Inline comments carry a fingerprint, so they're never posted twice. PRs from forks fall back to annotations, because their token is read-only.

## 5. Evaluate

```bash
python eval/evaluate.py --data data/v2/valid.jsonl                       # rules baseline
python eval/evaluate.py --data data/v2/valid.jsonl --backend llamacpp --model bunnyweb/tidypr-1.5b-gguf --no-rules
```

The validation set is synthetic, so the rules look better on it than they will on real code. The numbers that count come from the hand-labelled gold set (`data/gold/`, next milestone).

## Layout

| Path | What |
|---|---|
| `tidypr/diff.py` | unified diff → hunks with new-file line numbers |
| `tidypr/rules.py` | zero-dep rule pass |
| `tidypr/prompt.py` | the one prompt format used by training **and** inference |
| `tidypr/schema.py` | categories, `Finding`, JSON schema, output parser |
| `tidypr/synth.py` | slop injectors for synthetic data |
| `tidypr/infer.py` | backends (`rules`, `llamacpp`, `mlx`) and the review pipeline |
| `tidypr/github.py` | summary comment, review with suggestions, annotations fallback |
| `training/` | Unsloth, MLX, GGUF export, HF push, Kaggle notebook |
| `eval/evaluate.py` | per-category P/R/F1 and the false-positive gate |
| `docs/PLAN.md` | the full plan |
