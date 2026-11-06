"""LoRA fine-tune of a small Qwen3 model on a Tironian dataset → one quantized GGUF.

Runs on a CUDA GPU: inside Modal (``modal_app.py``) or on the user's own machine (driver
``--local``). It is deliberately a single, boring script so that both paths are the same code.

Prompt parity with serving: llama.cpp renders the GGUF's Qwen3 chat template with
``enable_thinking=False``, which produces ChatML and opens the reply with an empty think block.
Training prompts are rendered the same way, and the loss covers only the completion (the
teacher's reply), so the student learns to answer, never to imitate the prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROLE = {"system": "system", "human": "user", "tool": "user", "gpt": "assistant"}
GEN_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
_THINK = re.compile(r"^\s*<think>[\s\S]*?</think>\s*")


def strip_think(text: str) -> str:
    return _THINK.sub("", text or "", count=1)


def render_example(conversations: list[dict[str, str]]) -> tuple[str, str]:
    """→ (prompt, completion) in Qwen3 ChatML, identical to what llama.cpp serves."""
    *context, final = conversations
    if final.get("from") != "gpt":
        raise ValueError("last turn must be the teacher's reply")
    parts = []
    for turn in context:
        role = ROLE.get(turn["from"], "user")
        content = strip_think(turn["value"]) if role == "assistant" else turn["value"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    prompt = "".join(parts) + GEN_PREFIX
    completion = strip_think(final["value"]).strip() + "<|im_end|>"
    return prompt, completion


def load_examples(path: Path) -> list[dict[str, str]]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                p, c = render_example(json.loads(line)["conversations"])
                out.append({"prompt": p, "completion": c})
    return out


class Progress:
    def __init__(self, path: Path | None, sink=None) -> None:
        self.path, self.sink = path, sink

    def __call__(self, state: str, progress: float, message: str, **extra) -> None:
        rec = {"t": time.time(), "state": state, "progress": round(progress, 4), "message": message, **extra}
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        if self.sink:
            try:
                self.sink(rec)
            except Exception:
                pass
        print(json.dumps(rec), flush=True)


def run(data: Path, base: str, out_dir: Path, *, epochs: float = 2.0, seed: int = 3407, lr: float = 2e-4,
        rank: int = 16, max_seq: int = 4096, quant: str = "q4_k_m", name: str = "student",
        progress: Progress | None = None) -> dict:
    progress = progress or Progress(None)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = load_examples(data)
    if len(examples) < 8:
        raise RuntimeError(f"only {len(examples)} training examples; need at least 8")
    progress("loading", 0.02, f"loading {base}")

    from unsloth import FastLanguageModel  # noqa: E402  (heavy; import after arg checks)
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(model_name=base, max_seq_length=max_seq,
                                                         load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=rank, lora_alpha=rank * 2, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=seed)

    class Cb(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if state.max_steps:
                frac = 0.05 + 0.75 * state.global_step / state.max_steps
                loss = (logs or {}).get("loss")
                progress("training", frac, f"step {state.global_step}/{state.max_steps}"
                         + (f" · loss {loss:.3f}" if isinstance(loss, (int, float)) else ""),
                         step=state.global_step, max_steps=state.max_steps, loss=loss)

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=Dataset.from_list(examples),
        callbacks=[Cb()],
        args=SFTConfig(
            output_dir=str(out_dir / "ckpt"), per_device_train_batch_size=2, gradient_accumulation_steps=4,
            num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
            logging_steps=2, save_strategy="no", seed=seed, max_length=max_seq, report_to="none",
            completion_only_loss=True, bf16=True))
    progress("training", 0.05, f"{len(examples)} examples · {epochs:g} epochs")
    stats = trainer.train()

    progress("exporting", 0.82, f"merging and quantizing to {quant.upper()} GGUF")
    gguf_dir = out_dir / "gguf"
    model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=quant)
    # Unsloth has written into the folder it was given and, in newer releases, into a sibling
    # "<folder>_gguf". Look in both, and anywhere else under the output folder.
    found = {p for d in (gguf_dir, Path(f"{gguf_dir}_gguf"), out_dir) if d.exists() for p in d.rglob("*.gguf")}
    ggufs = sorted((g for g in found if "mmproj" not in g.name.lower()), key=lambda p: p.stat().st_size, reverse=True)
    # A full-precision copy may sit next to the quantized one; the quantized one is the student.
    ggufs = [g for g in ggufs if quant.upper() in g.name.upper()] or ggufs
    if not ggufs:
        seen = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file())[:40]
        raise RuntimeError(f"GGUF export produced no file; output folder holds: {seen}")
    final = out_dir / f"{name}.gguf"
    os.replace(ggufs[0], final)
    h = hashlib.sha256()
    with final.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    result = {"gguf": str(final), "sha256": h.hexdigest(), "size_bytes": final.stat().st_size,
              "train_loss": getattr(stats, "training_loss", None), "examples": len(examples), "base": base}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    progress("exported", 0.95, f"{final.name} · {result['size_bytes'] / 1e9:.2f} GB", **result)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", default="student")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--progress-file", type=Path)
    a = ap.parse_args(argv)
    run(a.data, a.base, a.out, epochs=a.epochs, seed=a.seed, name=a.name, progress=Progress(a.progress_file))
    return 0


if __name__ == "__main__":
    sys.exit(main())
