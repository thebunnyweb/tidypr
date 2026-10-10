"""Convert a fused HF-format model (e.g. from MLX --fuse) to a quantized GGUF with llama.cpp.

Needs a llama.cpp checkout (for convert_hf_to_gguf.py) and the llama-quantize binary:
  git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
  pip install -r ~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
  brew install llama.cpp        # provides llama-quantize on macOS

  python training/export_gguf.py --hf-dir runs/mlx/fused --out runs/mlx/slop-check-Q4_K_M.gguf

(The Unsloth path does this itself via save_pretrained_gguf.)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP_DIR", str(Path.home() / "llama.cpp")))
    a = ap.parse_args()

    f16 = Path(a.out).with_suffix(".f16.gguf")
    convert = Path(a.llama_cpp) / "convert_hf_to_gguf.py"
    if not convert.exists():
        sys.exit(f"missing {convert}; clone llama.cpp or set LLAMA_CPP_DIR")
    subprocess.run([sys.executable, str(convert), a.hf_dir, "--outfile", str(f16), "--outtype", "f16"], check=True)

    quantize = shutil.which("llama-quantize") or str(Path(a.llama_cpp) / "build/bin/llama-quantize")
    subprocess.run([quantize, str(f16), a.out, a.quant], check=True)
    f16.unlink(missing_ok=True)
    print(f"GGUF → {a.out}")


if __name__ == "__main__":
    main()
