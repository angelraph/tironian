"""Hermes function-calling format: one code path for training data *and* live student calls.

The single most common way a distilled model fails is a mismatch between the prompt it was
trained on and the prompt it is served. Tironian removes that failure mode by construction:
:func:`compact_messages` builds the student's context for both the training set and the live
router, and :func:`render_target` / :func:`parse_output` are exact inverses.

The format follows Hermes Agent's trajectory spec (``website/docs/developer-guide/
trajectory-format.md``): tools listed inside ``<tools>`` in the system prompt, calls emitted as
``<tool_call>{"name", "arguments"}</tool_call>``, results returned in ``<tool_response>``
blocks, and every assistant turn opening with a (possibly empty) ``<think>`` block.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

SYSTEM_HEADER = (
    "You are a function calling AI model. You are provided with function signatures within "
    "<tools> </tools> XML tags. You may call one or more functions to assist with the user query. "
    "If available tools are not relevant in assisting with user query, just respond in natural "
    "conversational language. Don't make assumptions about what values to plug into functions. "
    "After calling & executing the functions, you will be provided with function results within "
    "<tool_response> </tool_response> XML tags. Here are the available tools:\n<tools>\n{tools}\n</tools>\n"
    "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
    "{{'title': 'FunctionCall', 'type': 'object', 'properties': {{'name': {{'title': 'Name', 'type': 'string'}}, "
    "'arguments': {{'title': 'Arguments', 'type': 'object'}}}}, 'required': ['name', 'arguments']}}\n"
    "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
    "Example:\n<tool_call>\n{{'name': <function-name>,'arguments': <args-dict>}}\n</tool_call>"
)

_THINK = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)
_OPEN_THINK = re.compile(r"^\s*<think>[\s\S]*$", re.IGNORECASE)  # unterminated think = no answer
_TOOL_CALL = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.IGNORECASE)
_STRAY_TAG = re.compile(r"</?(?:tool_call|tool_response|tools|think)>", re.IGNORECASE)

CHARS_PER_TOKEN = 3.6


def est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


# Tools

def tool_name(tool: dict[str, Any]) -> str:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return str(fn.get("name") or "")


def tool_signature(tool: dict[str, Any]) -> dict[str, Any]:
    """The compact signature shown to the student (name, description, parameters)."""
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    desc = str(fn.get("description") or "")
    if len(desc) > 600:
        desc = desc[:600].rsplit(" ", 1)[0] + " …"
    return {"name": fn.get("name"), "description": desc,
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            "required": (fn.get("parameters") or {}).get("required")}


def render_system(tools: Sequence[dict[str, Any]]) -> str:
    sigs = [tool_signature(t) for t in tools]
    return SYSTEM_HEADER.format(tools=json.dumps(sigs, ensure_ascii=False))


def select_tools(tools: Sequence[dict[str, Any]], names: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(names)
    return [t for t in tools if tool_name(t) in wanted]


# Messages

def _text_of(content: Any) -> tuple[str, bool]:
    """Flatten OpenAI content (str or parts). Returns (text, had_image)."""
    if content is None:
        return "", False
    if isinstance(content, str):
        return content, False
    had_image = False
    parts: list[str] = []
    if isinstance(content, list):
        for p in content:
            if isinstance(p, dict):
                if p.get("type") in ("text", "input_text", "output_text"):
                    parts.append(str(p.get("text") or ""))
                elif p.get("type") in ("image_url", "input_image", "image"):
                    had_image = True
            elif isinstance(p, str):
                parts.append(p)
    return "\n".join(x for x in parts if x), had_image


def _args_obj(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return {}
    return {}


def render_tool_calls(tool_calls: Sequence[dict[str, Any]]) -> str:
    blocks = []
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        blocks.append("<tool_call>\n" + json.dumps(
            {"name": fn.get("name"), "arguments": _args_obj(fn.get("arguments"))}, ensure_ascii=False)
            + "\n</tool_call>")
    return "\n".join(blocks)


def render_target(content: str | None, tool_calls: Sequence[dict[str, Any]] | None) -> str:
    """Assistant turn text exactly as the student should produce it."""
    body = (content or "").strip()
    calls = render_tool_calls(tool_calls or [])
    joined = "\n".join(x for x in (body, calls) if x)
    return "<think>\n</think>\n" + joined


@dataclass
class Compact:
    messages: list[dict[str, str]]
    had_image: bool = False
    truncated: bool = False
    tokens: int = 0
    tool_names: list[str] = field(default_factory=list)


def compact_messages(messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]], *,
                     max_tokens: int = 3072) -> Compact:
    """Student context: Hermes tool prompt + the most recent turns that fit the budget.

    The original Hermes system prompt is replaced (it is long and generic); tool results are
    grouped into one ``<tool_response>`` user message, matching the trajectory spec.
    """
    system = render_system(tools)
    budget = max_tokens - est_tokens(system)
    had_image = False
    converted: list[dict[str, str]] = []
    pending_tool: list[str] = []

    def flush_tools() -> None:
        if pending_tool:
            converted.append({"role": "user", "content": "\n".join(pending_tool)})
            pending_tool.clear()

    for m in messages:
        role = m.get("role")
        if role == "system" or role == "developer":
            continue
        if role == "tool":
            text, _ = _text_of(m.get("content"))
            if len(text) > 4000:
                text = text[:2000] + "\n…[truncated]…\n" + text[-1500:]
            pending_tool.append("<tool_response>\n" + json.dumps(
                {"tool_call_id": m.get("tool_call_id"), "name": m.get("name"), "content": text},
                ensure_ascii=False) + "\n</tool_response>")
            continue
        flush_tools()
        text, img = _text_of(m.get("content"))
        had_image = had_image or img
        if role == "assistant":
            converted.append({"role": "assistant", "content": render_target(text, m.get("tool_calls"))})
        elif role == "user":
            converted.append({"role": "user", "content": text})
    flush_tools()

    # Keep the newest messages that fit; never start the window on an assistant/tool message.
    kept: list[dict[str, str]] = []
    used = 0
    for msg in reversed(converted):
        cost = est_tokens(msg["content"]) + 4
        if used + cost > budget and kept:
            break
        kept.append(msg)
        used += cost
    kept.reverse()
    truncated = len(kept) < len(converted)
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
        truncated = True
    # A leading user message that is really a tool response has lost its call; drop it too.
    while kept and kept[0]["content"].startswith("<tool_response>"):
        kept.pop(0)
        truncated = True
    msgs = [{"role": "system", "content": system}] + kept
    return Compact(msgs, had_image, truncated, used + est_tokens(system), [tool_name(t) for t in tools])


# Parsing student output

@dataclass
class Parsed:
    ok: bool
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)   # [{"name", "arguments"}]
    error: str | None = None


# A backslash not starting a legal JSON escape (and not itself escaped).
_BAD_ESCAPE = re.compile(r'(?<!\\)\\(?![\\"/bfnrtu])')


def parse_output(text: str) -> Parsed:
    if text is None:
        return Parsed(False, error="empty")
    raw = str(text)
    if _OPEN_THINK.match(raw) and "</think>" not in raw.lower():
        return Parsed(False, error="unterminated_think")
    body = _THINK.sub("", raw).strip()
    calls: list[dict[str, Any]] = []
    for m in _TOOL_CALL.finditer(body):
        blob = m.group(1).strip()
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            # Found live: small models write shell escapes like `-iname \*.docx` inside a JSON
            # string, which isn't a legal JSON escape. Keeping the backslash literally gives the
            # exact command the model wrote; anything else broken still fails.
            try:
                obj = json.loads(_BAD_ESCAPE.sub(r"\\\\", blob))
            except json.JSONDecodeError:
                return Parsed(False, error="tool_call_json")
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            return Parsed(False, error="tool_call_shape")
        args = obj.get("arguments", {})
        if isinstance(args, str):
            args = _args_obj(args)
        if not isinstance(args, dict):
            return Parsed(False, error="tool_call_args")
        calls.append({"name": obj["name"], "arguments": args})
    content = _TOOL_CALL.sub("", body).strip()
    if _STRAY_TAG.search(content):
        return Parsed(False, error="stray_tag")
    if not calls and not content:
        return Parsed(False, error="empty")
    return Parsed(True, content, calls)


# Minimal JSON-schema validation for tool arguments

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
    "object": (dict,), "array": (list,), "null": (type(None),),
}


def validate_args(args: dict[str, Any], schema: dict[str, Any] | None) -> str | None:
    """Return None if valid, else a short reason. Covers what tool schemas actually use."""
    if not schema:
        return None
    props: dict[str, Any] = schema.get("properties") or {}
    for req in schema.get("required") or []:
        if req not in args:
            return f"missing:{req}"
    if schema.get("additionalProperties") is False:
        extra = set(args) - set(props)
        if extra:
            return f"unexpected:{sorted(extra)[0]}"
    for key, val in args.items():
        spec = props.get(key)
        if not isinstance(spec, dict):
            continue
        types = spec.get("type")
        if types:
            names = types if isinstance(types, list) else [types]
            allowed: tuple[type, ...] = tuple(t for n in names for t in _TYPES.get(n, (object,)))
            if isinstance(val, bool) and "boolean" not in names and bool not in allowed:
                return f"type:{key}"
            if not isinstance(val, allowed):
                return f"type:{key}"
        if "enum" in spec and val not in spec["enum"]:
            return f"enum:{key}"
    return None


def tool_schema_map(tools: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for t in tools:
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        out[str(fn.get("name"))] = fn.get("parameters") or {}
    return out


# ShareGPT export

_ROLE_TO_FROM = {"system": "system", "user": "human", "assistant": "gpt"}


def to_sharegpt(compact: Compact, target: str) -> list[dict[str, str]]:
    """Training conversation: compact context + the teacher's response as the final gpt turn.

    User messages that are tool responses become ``tool`` turns, as in the Hermes spec.
    """
    convo = []
    for m in compact.messages:
        frm = _ROLE_TO_FROM.get(m["role"], "human")
        if frm == "human" and m["content"].startswith("<tool_response>"):
            frm = "tool"
        convo.append({"from": frm, "value": m["content"]})
    convo.append({"from": "gpt", "value": target})
    return convo
