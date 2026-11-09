"""Typed view of Tironian's settings (``plugins.entries.tironian.settings``).

Defaults here mirror ``config_schema`` in ``plugin.yaml``; a unit test keeps them in sync.
Every read goes through :meth:`Settings.load` so a bad value from the Desktop form can never
crash a hook: unknown or mistyped values fall back to the default.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable

PLUGIN_ID = "tironian"


@dataclass
class Settings:
    capture_enabled: bool = True
    routing_enabled: bool = False
    promotion_mode: str = "shadow"            # shadow | live (what a gate pass promotes to)
    exclude_platforms: tuple[str, ...] = ("cron", "kanban")
    capture_subagents: bool = False
    redact_pii: bool = True
    retention_days: int = 90
    judge_enabled: bool = False
    judge_daily_budget: int = 50
    base_model: str = "qwen3-1.7b"            # qwen3-1.7b | qwen3-4b
    train_backend: str = "modal"              # modal | local
    modal_gpu: str = "L4"                     # L4 | A10G
    max_usd_per_run: float = 1.0
    promotion_threshold: float = 0.85
    spot_check_rate: float = 0.1              # share of a live family's requests double-checked for drift
    answer_badge: str = "desktop"             # desktop | always | off: mark replies your own model helped write
    min_eval_n: int = 30
    validity_floor: float = 0.98
    min_family_turns: int = 25
    route_confidence_min: float = 0.55
    route_margin_min: float = 0.10
    student_endpoint: str = "http://127.0.0.1:8080/v1"
    student_timeout_s: int = 25
    max_context_tokens: int = 3072
    teacher_usd_per_mtok_in: float = 3.0
    teacher_usd_per_mtok_out: float = 15.0

    @classmethod
    def load(cls, get: Callable[..., Any]) -> "Settings":
        s = cls()
        for f in fields(cls):
            default = getattr(s, f.name)
            try:
                raw = get(f.name, default=default)
            except Exception:
                raw = default
            setattr(s, f.name, _coerce(raw, default))
        if s.promotion_mode not in ("shadow", "live"):
            s.promotion_mode = "shadow"
        if s.train_backend not in ("modal", "local"):
            s.train_backend = "modal"
        if s.base_model not in BASE_MODELS:
            s.base_model = "qwen3-1.7b"
        return s


def _coerce(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    try:
        if isinstance(default, bool):
            if isinstance(raw, str):
                return raw.strip().lower() in ("1", "true", "yes", "on")
            return bool(raw)
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        if isinstance(default, tuple):
            if isinstance(raw, str):
                return tuple(x.strip() for x in raw.split(",") if x.strip())
            return tuple(str(x) for x in raw)
        return str(raw)
    except (TypeError, ValueError):
        return default


#: Base models a student can be trained from (Hugging Face ids used by the trainer).
BASE_MODELS = {
    "qwen3-1.7b": "unsloth/Qwen3-1.7B",
    "qwen3-4b": "unsloth/Qwen3-4B",
}

#: Approximate per-second GPU prices on Modal (USD), used for estimates and the cost cap.
MODAL_GPU_USD_PER_S = {"L4": 0.000222, "A10G": 0.000306}
