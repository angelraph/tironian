"""Getting a trained student in front of a local model server, without asking the user to.

Two servers are supported out of the box:

- Hermes's own llama.cpp runtime (``http://127.0.0.1:8080/v1``), when it's installed.
- Ollama (``http://127.0.0.1:11434/v1``), which many people already run. Tironian registers the
  student with ``ollama create`` and a Modelfile whose template matches training exactly:
  Qwen3 ChatML with an empty, closed think block before the reply.

With the default endpoint, Tironian uses whichever of the two lists the student.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

HERMES_ENDPOINT = "http://127.0.0.1:8080/v1"
OLLAMA_ENDPOINT = "http://127.0.0.1:11434/v1"

MODELFILE = '''FROM ./{gguf}
TEMPLATE """{{{{- range .Messages }}}}<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
<think>

</think>

"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER num_ctx 4096
PARAMETER temperature 0
'''


def find_ollama() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    return str(local) if local.exists() else None


def load_into_ollama(gguf: Path, name: str, *, timeout_s: int = 900) -> tuple[bool, str]:
    """Register ``gguf`` with Ollama as ``name``. Safe to repeat: Ollama reuses the same blob."""
    exe = find_ollama()
    if not exe:
        return False, "Ollama isn't installed"
    if not gguf.exists():
        return False, f"model file missing: {gguf}"
    mf = gguf.parent / f"Modelfile.{name}"
    mf.write_text(MODELFILE.format(gguf=gguf.name), encoding="utf-8")
    try:
        out = subprocess.run([exe, "create", name, "-f", str(mf)], cwd=str(gguf.parent), capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"ollama create failed: {exc}"
    if out.returncode != 0:
        return False, f"ollama create failed: {(out.stderr or out.stdout).strip()[-300:]}"
    return True, f"loaded into Ollama as {name}"


def _lists(endpoint: str, model_id: str | None) -> bool:
    from .student import StudentClient, model_matches
    models = StudentClient(endpoint).available_models()
    if model_id is None:
        return any(m.lower().startswith("tironian-") for m in models)
    return any(model_matches(m, model_id) or model_matches(m.split(":")[0], model_id) for m in models)


_cache: dict[str, tuple[float, str]] = {}
_lock = threading.Lock()


def resolve_endpoint(configured: str, model_id: str | None = None, *, ttl_s: float = 60.0) -> str:
    """The configured endpoint, unless it's the default and only Ollama has a student.

    A user who set their own endpoint always gets exactly that one.
    """
    if configured.rstrip("/") != HERMES_ENDPOINT:
        return configured
    key = model_id or "*"
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl_s:
            return hit[1]
    chosen = configured
    if not _lists(HERMES_ENDPOINT, model_id) and _lists(OLLAMA_ENDPOINT, model_id):
        chosen = OLLAMA_ENDPOINT
    with _lock:
        _cache[key] = (now, chosen)
    return chosen


def forget() -> None:
    with _lock:
        _cache.clear()
