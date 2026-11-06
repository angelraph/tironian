"""Shared fixtures: load the plugin as a package named ``tironian`` and fake Hermes pieces."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent / "plugin"


def _load_package() -> Any:
    if "tironian" in sys.modules:
        return sys.modules["tironian"]
    spec = importlib.util.spec_from_file_location("tironian", ROOT / "__init__.py",
                                                  submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tironian"] = mod
    spec.loader.exec_module(mod)
    return mod


_load_package()


class FakeCtx:
    """Records what the plugin registers, like Hermes's PluginContext."""

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self.hooks: dict[str, list] = {}
        self.middleware: dict[str, list] = {}
        self.commands: dict[str, Any] = {}
        self.cli: dict[str, Any] = {}
        self.aux: list[str] = []
        self.settings = settings or {}
        self.llm = None

    def register_hook(self, name, fn):
        self.hooks.setdefault(name, []).append(fn)

    def register_middleware(self, kind, fn):
        self.middleware.setdefault(kind, []).append(fn)

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = handler

    def register_cli_command(self, name, help, setup_fn, handler_fn):
        self.cli[name] = (setup_fn, handler_fn)

    def register_auxiliary_task(self, name, **kw):
        self.aux.append(name)

    def get_config(self, key, default=None):
        return self.settings.get(key, default)


class FakeLlama:
    """An OpenAI-compatible server whose next reply the test controls."""

    def __init__(self) -> None:
        self.replies: list[Any] = []
        self.model = "tironian-fam-v1"
        self.calls: list[dict[str, Any]] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({"data": [{"id": outer.model}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                outer.calls.append(json.loads(self.rfile.read(n)))
                reply = outer.replies.pop(0) if outer.replies else "<think>\n</think>\nok"
                if isinstance(reply, int):
                    self.send_response(reply)
                    self.end_headers()
                    return
                body = json.dumps({"model": outer.model, "choices": [{"message": {"content": reply},
                                   "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def llama():
    srv = FakeLlama()
    yield srv
    srv.close()


@pytest.fixture
def app_factory(tmp_path):
    from tironian.core.app import App

    made = []

    def make(async_capture=False, **settings):
        a = App(data_dir=tmp_path / "plugin-data" / "tironian", config_get=lambda k, default=None: settings.get(k, default),
                secret=lambda n: None, broadcast=lambda k, p: None)
        a.engine.sync_capture = not async_capture
        made.append(a)
        return a

    yield make
    for a in made:
        a.store.close()
