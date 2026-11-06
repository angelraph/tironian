"""Tironian: your own small model, learned from your own Hermes work.

Tironian captures every model step Hermes takes (redacted before storage), grades each turn
from what actually happened, clusters recurring tasks into families, fine-tunes a small local
model per family, proves it on held-out steps from your own sessions, and only then lets it
answer matching requests, checking every answer and falling back to your model silently.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

log = logging.getLogger("tironian")

_HOOKS = (
    "on_session_start", "pre_llm_call", "post_llm_call", "post_tool_call", "api_request_error",
    "on_session_end", "on_session_finalize", "on_session_delete", "post_approval_response", "subagent_start",
)


def _safe(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(**kwargs: Any) -> None:
        try:
            fn(**kwargs)
        except Exception:  # hooks are fail-open in Hermes too; never let capture disturb a turn
            log.debug("tironian hook %s failed", getattr(fn, "__name__", fn), exc_info=True)
        return None
    wrapper.__name__ = f"tironian_{getattr(fn, '__name__', 'hook')}"
    return wrapper


def register(ctx: Any) -> None:
    # Keep registration cheap: Hermes loads plugins on every start. Heavier modules are
    # imported on first use (first hook call, first /tironian, first hermes tironian).
    from .core import cli_args

    state: dict[str, Any] = {}

    def app() -> Any:
        # Lazy: the database opens on first use, never during plugin discovery.
        if "app" not in state:
            from .core.app import get_app
            state["app"] = get_app(config_get=lambda k, default=None: ctx.get_config(k, default=default),
                                   llm=getattr(ctx, "llm", None))
        return state["app"]

    def hook(name: str) -> Callable[..., Any]:
        def call(**kw: Any) -> None:
            a = app()
            if name == "on_session_start":
                a.first_run()
                a.engine.on_session_start(**kw)
            elif name == "subagent_start":
                a.engine.on_subagent_start(**kw)
            else:
                getattr(a.engine, name)(**kw)
        call.__name__ = name
        return _safe(call)

    for name in _HOOKS:
        ctx.register_hook(name, hook(name))

    def llm_execution(**kw: Any) -> Any:
        return app().engine.llm_execution(**kw)

    ctx.register_middleware("llm_execution", llm_execution)

    def slash(raw: str, *, session_id: str = "", platform: str = "", **kw: Any) -> str:
        from .core import commands
        a = app()
        a.first_run()
        ask = getattr(ctx, "ask_choice", None)
        return commands.handle(a, raw, session_id=session_id, platform=platform, ask=ask)

    ctx.register_command("tironian", handler=slash, description="Your personal model: status, report, train, promote",
                         args_hint="[status|report|families|good|bad|off|on|train|shadow|promote|demote|why]")
    ctx.register_cli_command(name="tironian", help="Distill your Hermes work into a personal local model",
                             setup_fn=cli_args.setup, handler_fn=cli_args.run)

    register_aux = getattr(ctx, "register_auxiliary_task", None)
    if callable(register_aux):
        register_aux("tironian_judge", display_name="Tironian judge",
                     description="Blind A/B judge for held-out evaluation of trained students (optional, budgeted).")
