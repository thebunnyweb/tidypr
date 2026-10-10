"""LoRA fine-tune on Apple Silicon with MLX (local Mac path). Same data as the Unsloth path.

  pip install -e ".[train-mlx]"
  python training/train_mlx.py --data data/out --out runs/mlx            # 0.5B, fits 8–16 GB Macs
  python training/train_mlx.py --base Qwen/Qwen2.5-Coder-1.5B-Instruct   # 16 GB+

Then:  python training/export_gguf.py --mlx runs/mlx  →  GGUF  →  training/push_to_hub.py
Memory tips: lower --batch to 1, --num-layers to 8, --max-seq to 2048.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/out", help="dir with train.jsonl + valid.jsonl")
    ap.add_argument("--out", default="runs/mlx")
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--num-layers", type=int, default=16, help="how many top layers get LoRA")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq", type=int, default=3072)
    ap.add_argument("--fuse", action="store_true", help="also fuse adapter into a full HF-format model")
    a = ap.parse_args()

    adapters = f"{a.out}/adapters"
    run([sys.executable, "-m", "mlx_lm.lora", "--model", a.base, "--train", "--data", a.data,
         "--iters", str(a.iters), "--batch-size", str(a.batch), "--num-layers", str(a.num_layers),
         "--learning-rate", str(a.lr), "--max-seq-length", str(a.max_seq), "--mask-prompt",
         "--grad-checkpoint", "--steps-per-eval", "100", "--save-every", "200", "--adapter-path", adapters])
    print(f"\nadapter → {adapters}\ntry it:  slopcheck review --diff data/seed/sample_pr.diff "
          f"--backend mlx --model {a.base} --adapter {adapters}")

    if a.fuse:
        # --dequantize gives plain fp16 safetensors that llama.cpp's converter understands.
        run([sys.executable, "-m", "mlx_lm.fuse", "--model", a.base, "--adapter-path", adapters,
             "--save-path", f"{a.out}/fused", "--dequantize"])
        print(f"fused HF model → {a.out}/fused  (next: training/export_gguf.py --hf-dir {a.out}/fused)")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
