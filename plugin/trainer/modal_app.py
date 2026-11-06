"""Modal app that runs ``train_core`` on a rented GPU.

Only the driver (in Tironian's sidecar venv) imports this. Data moves through a private
Modal Volume in the user's own Modal workspace (``tironian``): the dataset is uploaded for
the job and the trained GGUF is downloaded back, then both are deleted from the volume.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

HERE = Path(__file__).parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .pip_install_from_requirements(str(HERE / "requirements-train.txt"))
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(HERE / "train_core.py"), "/root/train_core.py")
)

app = modal.App("tironian-trainer", image=image)
volume = modal.Volume.from_name("tironian", create_if_missing=True)
progress_dict = modal.Dict.from_name("tironian-progress", create_if_missing=True)
cache = modal.Volume.from_name("tironian-hf-cache", create_if_missing=True)


# GPU and timeout come from the driver (cost cap = timeout × GPU price); defaults apply in-container.
@app.function(gpu=os.environ.get("TIRONIAN_GPU", "L4"), timeout=int(os.environ.get("TIRONIAN_TIMEOUT", "3600")), volumes={"/vol": volume, "/root/.cache/huggingface": cache})
def train(job_id: str, base: str, name: str, epochs: float = 2.0, seed: int = 3407) -> dict:
    import sys

    sys.path.insert(0, "/root")
    import train_core

    job = Path("/vol/jobs") / job_id

    def sink(rec: dict) -> None:
        progress_dict[job_id] = rec

    prog = train_core.Progress(job / "progress.jsonl", sink)
    try:
        result = train_core.run(job / "train.jsonl", base, job / "out", epochs=epochs, seed=seed, name=name,
                                progress=prog)
    except BaseException as exc:
        prog("failed", 1.0, f"{type(exc).__name__}: {exc}")
        raise
    finally:
        volume.commit()
        cache.commit()
    result["remote_gguf"] = f"/jobs/{job_id}/out/{name}.gguf"
    return result
