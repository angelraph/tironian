"""Calling a student model and shaping its answer like a provider response.

Default backend: the OpenAI-compatible endpoint of Hermes's managed llama.cpp runtime on
localhost (``student_endpoint``). That keeps routing deterministic and private, needs no
trust-gated provider override, and lets Tironian check that the answer really came from the
student. Any OpenAI-compatible server (Ollama, LM Studio, a remote box the user owns) works.

Only the standard library is used; the response object is built with ``openai.types`` when
available (it ships with Hermes) and a faithful attribute-access stand-in otherwise.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence


@dataclass
class StudentReply:
    ok: bool
    text: str = ""
    model: str = ""
    latency_ms: int = 0
    in_tok: int = 0
    out_tok: int = 0
    error: str | None = None


class StudentClient:
    def __init__(self, endpoint: str, *, timeout_s: float = 25.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/v1"):
            self.endpoint += "/v1"
        self.timeout_s = timeout_s

    def complete(self, model_id: str, messages: Sequence[dict[str, str]], *,
                 max_tokens: int = 768, temperature: float = 0.0) -> StudentReply:
        import urllib.error
        import urllib.request

        body = json.dumps({
            "model": model_id, "messages": list(messages), "temperature": temperature,
            "max_tokens": max_tokens, "stream": False,
            # llama.cpp's Qwen3 template: answer directly, no hidden reasoning pass.
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode("utf-8")
        req = urllib.request.Request(self.endpoint + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 - user-set local endpoint
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return StudentReply(False, latency_ms=int((time.monotonic() - started) * 1000),
                                error=f"unreachable:{type(exc).__name__}")
        except (ValueError, json.JSONDecodeError):
            return StudentReply(False, latency_ms=int((time.monotonic() - started) * 1000), error="bad_json")
        latency = int((time.monotonic() - started) * 1000)
        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
            if choice.get("finish_reason") == "length":
                return StudentReply(False, text, data.get("model", ""), latency, error="truncated")
        except (KeyError, IndexError, TypeError, AttributeError):
            return StudentReply(False, latency_ms=latency, error="bad_shape")
        usage = data.get("usage") or {}
        return StudentReply(True, text, str(data.get("model") or ""), latency,
                            int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0))

    def available_models(self) -> list[str]:
        import urllib.request

        try:
            with urllib.request.urlopen(self.endpoint + "/models", timeout=3) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
            return [str(m.get("id")) for m in data.get("data", []) if m.get("id")]
        except Exception:
            return []


def model_matches(served: str, wanted: str) -> bool:
    """llama.cpp may echo a path or alias; accept exact id or a path ending in it."""
    if not served:
        return True  # some servers omit the field; the endpoint itself is the student's
    s = served.replace("\\", "/").lower()
    w = wanted.lower()
    return s == w or s.endswith("/" + w) or s.endswith("/" + w + ".gguf") or s == w + ".gguf"


def to_chat_completion(content: str, tool_calls: Sequence[dict[str, Any]], *, model: str,
                       in_tok: int, out_tok: int) -> Any:
    """Shape a validated student step exactly like an OpenAI ``ChatCompletion``."""
    calls = [{
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)},
    } for c in tool_calls]
    payload = {
        "id": f"chatcmpl-tironian-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls" if calls else "stop",
            "message": {"role": "assistant", "content": content or None, "tool_calls": calls or None},
        }],
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok, "total_tokens": in_tok + out_tok},
    }
    try:
        from openai.types.chat import ChatCompletion  # shipped with Hermes
        return ChatCompletion.model_validate(payload)
    except Exception:
        return _ns(payload)


def _ns(x: Any) -> Any:
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_ns(v) for v in x]
    return x


def response_to_dict(resp: Any) -> dict[str, Any] | None:
    """Best-effort read of a provider response (OpenAI object, dict or namespace)."""
    if resp is None:
        return None
    if isinstance(resp, dict):
        return resp
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(resp, attr, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, dict):
                    return out
            except Exception:
                pass
    try:
        choices = getattr(resp, "choices", None) or []
        msg = choices[0].message if choices else None
        calls = []
        for tc in getattr(msg, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            calls.append({"id": getattr(tc, "id", None), "type": "function",
                          "function": {"name": getattr(fn, "name", None), "arguments": getattr(fn, "arguments", None)}})
        usage = getattr(resp, "usage", None)
        return {
            "model": getattr(resp, "model", None),
            "choices": [{"finish_reason": getattr(choices[0], "finish_reason", None) if choices else None,
                         "message": {"role": "assistant", "content": getattr(msg, "content", None),
                                     "tool_calls": calls or None}}],
            "usage": {"prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                      "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                      "cost": getattr(usage, "cost", None)},
        }
    except Exception:
        return None
