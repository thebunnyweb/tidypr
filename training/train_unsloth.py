"""QLoRA fine-tune with Unsloth on an NVIDIA GPU (Kaggle / Colab / RunPod / local CUDA).

  pip install -e ".[train-unsloth]"
  python training/train_unsloth.py --data data/out --out runs/unsloth \
      --gguf q4_k_m --push bunnyweb/tidypr-1.5b-gguf     # push needs HF_TOKEN

Outputs in --out: lora/ (adapter), and with --gguf a .gguf file ready for llama.cpp.
"""
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/out")
    ap.add_argument("--out", default="runs/unsloth")
    ap.add_argument("--base", default="unsloth/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1, help="override epochs, e.g. 30 for a smoke test")
    ap.add_argument("--gguf", default=None, help="quantization to export, e.g. q4_k_m or q8_0")
    ap.add_argument("--push", default=None, help="HF repo id for the GGUF (uses HF_TOKEN)")
    a = ap.parse_args()

    # Unsloth must be imported before transformers/trl to apply its patches.
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(a.base, max_seq_length=a.max_seq, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=13,
    )

    ds = load_dataset("json", data_files={"train": f"{a.data}/train.jsonl", "valid": f"{a.data}/valid.jsonl"})
    ds = ds.map(lambda b: {"text": [tokenizer.apply_chat_template(m, tokenize=False) for m in b["messages"]]},
                batched=True, remove_columns=ds["train"].column_names)  # keep only "text" (drops "meta")

    cfg = dict(
        dataset_text_field="text", packing=False,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.01,
        logging_steps=10, eval_steps=100, save_steps=200, save_total_limit=2,
        fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),  # T4 has no bf16
        optim="adamw_8bit", seed=13, output_dir=f"{a.out}/checkpoints", report_to="none",
    )
    # trl renamed a few fields across versions; set whichever this install understands.
    fields = set(inspect.signature(SFTConfig.__init__).parameters)
    cfg["max_length" if "max_length" in fields else "max_seq_length"] = a.max_seq
    cfg["eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"] = "steps"

    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds["train"],
                         eval_dataset=ds["valid"], args=SFTConfig(**cfg))
    # Loss only on the assistant JSON, not on the prompt/diff (Qwen chat markers).
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                      response_part="<|im_start|>assistant\n")
    trainer.train()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(f"{a.out}/lora")
    tokenizer.save_pretrained(f"{a.out}/lora")
    print(f"saved adapter → {a.out}/lora")

    if a.gguf:
        model.save_pretrained_gguf(f"{a.out}/gguf", tokenizer, quantization_method=a.gguf)
        print(f"saved GGUF → {a.out}/gguf")
        if a.push:
            # Same uploader as the MLX path, so both produce identical HF repos.
            from push_to_hub import push
            gguf = pick_gguf(a.out, a.gguf)
            push(a.push, gguf, base=a.base, token=os.environ.get("HF_TOKEN"))


def pick_gguf(out: str, quant: str) -> Path:
    """Unsloth also leaves a large F16 .gguf around; pick the one we asked for (e.g. Q4_K_M)."""
    files = sorted(Path(out).rglob("*.gguf"))
    wanted = [f for f in files if quant.upper() in f.name.upper()]
    if not wanted:
        raise SystemExit(f"no {quant} .gguf under {out}; found: {[f.name for f in files]}")
    return wanted[0]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
