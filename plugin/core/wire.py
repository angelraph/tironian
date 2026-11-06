"""Normalise Hermes provider requests/responses into one OpenAI-chat shape.

Hermes talks to providers in three wire formats (``api_mode``): ``chat_completions``
(OpenAI-compatible: OpenRouter, Nous Portal, llama.cpp, …), ``anthropic_messages`` (Claude's
native API) and ``codex_responses`` (OpenAI Responses). Capture works for all three; Tironian
stores everything as OpenAI chat messages + function tools so one exporter and one student
format cover every teacher.
"""

from __future__ import annotations

import json
from typing import Any

from .student import response_to_dict


def _tools_from_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for t in tools or []:
        if not isinstance(t, dict) or "name" not in t:
            continue
        out.append({"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""),
            "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}})
    return out


def _tools_from_responses(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and "name" in t:
            out.append({"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("parameters") or {"type": "object", "properties": {}}}})
        elif isinstance(t.get("function"), dict):
            out.append(t)
    return out


def _anthropic_messages(req: dict[str, Any]) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    system = req.get("system")
    if system:
        text = system if isinstance(system, str) else "\n".join(
            b.get("text", "") for b in system if isinstance(b, dict))
        msgs.append({"role": "system", "content": text})
    for m in req.get("messages") or []:
        role, content = m.get("role"), m.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        texts, calls, results = [], [], []
        for b in content or []:
            if not isinstance(b, dict):
                continue
            kind = b.get("type")
            if kind == "text":
                texts.append(b.get("text", ""))
            elif kind == "tool_use":
                calls.append({"id": b.get("id"), "type": "function", "function": {
                    "name": b.get("name"), "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}})
            elif kind == "tool_result":
                rc = b.get("content")
                if isinstance(rc, list):
                    rc = "\n".join(x.get("text", "") for x in rc if isinstance(x, dict))
                results.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": rc or ""})
            elif kind == "image":
                texts.append("")  # marker handled by the image flag below
        if role == "assistant":
            msgs.append({"role": "assistant", "content": "\n".join(texts) or None, "tool_calls": calls or None})
        else:
            msgs.extend(results)
            if any(t for t in texts) or not results:
                has_image = any(isinstance(b, dict) and b.get("type") == "image" for b in content or [])
                parts: Any = "\n".join(texts)
                if has_image:
                    parts = [{"type": "text", "text": parts}, {"type": "image_url"}]
                msgs.append({"role": "user", "content": parts})
    return msgs


def _responses_messages(req: dict[str, Any]) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    if req.get("instructions"):
        msgs.append({"role": "system", "content": req["instructions"]})
    items = req.get("input")
    if isinstance(items, str):
        return msgs + [{"role": "user", "content": items}]
    pending_calls: list[dict[str, Any]] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        kind = it.get("type") or ("message" if "role" in it else None)
        if kind == "message":
            if pending_calls:
                msgs.append({"role": "assistant", "content": None, "tool_calls": pending_calls})
                pending_calls = []
            content = it.get("content")
            if isinstance(content, list):
                content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
            msgs.append({"role": it.get("role", "user"), "content": content})
        elif kind == "function_call":
            pending_calls.append({"id": it.get("call_id"), "type": "function", "function": {
                "name": it.get("name"), "arguments": it.get("arguments") or "{}"}})
        elif kind == "function_call_output":
            if pending_calls:
                msgs.append({"role": "assistant", "content": None, "tool_calls": pending_calls})
                pending_calls = []
            msgs.append({"role": "tool", "tool_call_id": it.get("call_id"), "content": str(it.get("output") or "")})
    if pending_calls:
        msgs.append({"role": "assistant", "content": None, "tool_calls": pending_calls})
    return msgs


def normalize_request(request: dict[str, Any], api_mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """→ (OpenAI chat messages, OpenAI function tools)."""
    if api_mode == "anthropic_messages":
        return _anthropic_messages(request), _tools_from_anthropic(request.get("tools") or [])
    if api_mode == "codex_responses":
        return _responses_messages(request), _tools_from_responses(request.get("tools") or [])
    tools = [t for t in (request.get("tools") or []) if isinstance(t, dict) and isinstance(t.get("function"), dict)]
    return list(request.get("messages") or []), tools


def normalize_response(resp: Any, api_mode: str) -> dict[str, Any] | None:
    """→ {"content", "tool_calls" (OpenAI shape), "finish_reason", "in_tok", "out_tok", "cost"}."""
    if api_mode == "anthropic_messages":
        d = response_to_dict(resp) or {}
        blocks = d.get("content") or []
        texts, calls = [], []
        for b in blocks if isinstance(blocks, list) else []:
            if isinstance(b, dict) and b.get("type") == "text":
                texts.append(b.get("text", ""))
            elif isinstance(b, dict) and b.get("type") == "tool_use":
                calls.append({"id": b.get("id"), "type": "function", "function": {
                    "name": b.get("name"), "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}})
        usage = d.get("usage") or {}
        return {"content": "\n".join(texts), "tool_calls": calls, "finish_reason": d.get("stop_reason"),
                "in_tok": int(usage.get("input_tokens") or 0), "out_tok": int(usage.get("output_tokens") or 0),
                "cost": None, "model": d.get("model")}
    if api_mode == "codex_responses":
        d = response_to_dict(resp) or {}
        texts, calls = [], []
        for it in d.get("output") or []:
            if not isinstance(it, dict):
                continue
            if it.get("type") == "message":
                for p in it.get("content") or []:
                    if isinstance(p, dict) and p.get("type") in ("output_text", "text"):
                        texts.append(p.get("text", ""))
            elif it.get("type") == "function_call":
                calls.append({"id": it.get("call_id"), "type": "function", "function": {
                    "name": it.get("name"), "arguments": it.get("arguments") or "{}"}})
        usage = d.get("usage") or {}
        return {"content": "\n".join(texts), "tool_calls": calls, "finish_reason": d.get("status"),
                "in_tok": int(usage.get("input_tokens") or 0), "out_tok": int(usage.get("output_tokens") or 0),
                "cost": None, "model": d.get("model")}
    d = response_to_dict(resp)
    if not d or not d.get("choices"):
        return None
    ch = d["choices"][0] or {}
    msg = ch.get("message") or {}
    usage = d.get("usage") or {}
    return {"content": msg.get("content") or "", "tool_calls": list(msg.get("tool_calls") or []),
            "finish_reason": ch.get("finish_reason"),
            "in_tok": int(usage.get("prompt_tokens") or 0), "out_tok": int(usage.get("completion_tokens") or 0),
            "cost": usage.get("cost"), "model": d.get("model")}


def build_response(api_mode: str, content: str, tool_calls: list[dict[str, Any]], *, model: str,
                   in_tok: int, out_tok: int) -> Any | None:
    """Provider-shaped response for a validated student step, or None if the mode is unsupported."""
    if api_mode == "chat_completions":
        from .student import to_chat_completion
        return to_chat_completion(content, tool_calls, model=model, in_tok=in_tok, out_tok=out_tok)
    if api_mode == "anthropic_messages":
        import uuid
        blocks: list[dict[str, Any]] = []
        if content:
            blocks.append({"type": "text", "text": content})
        for c in tool_calls:
            blocks.append({"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:24]}", "name": c["name"],
                           "input": c["arguments"]})
        payload = {"id": f"msg_tironian_{uuid.uuid4().hex[:16]}", "type": "message", "role": "assistant",
                   "model": model, "content": blocks,
                   "stop_reason": "tool_use" if tool_calls else "end_turn", "stop_sequence": None,
                   "usage": {"input_tokens": in_tok, "output_tokens": out_tok}}
        try:
            from anthropic.types import Message
            return Message.model_validate(payload)
        except Exception:
            return None  # without the real type, do not risk a shape Hermes cannot read
    return None
