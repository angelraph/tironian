"""register() must declare exactly what plugin.yaml says (catalog rule 6) and stay side-effect free."""

import re
from dataclasses import fields
from pathlib import Path

from conftest import FakeCtx

import tironian
from tironian.core.config import Settings

MANIFEST = (Path(__file__).resolve().parent.parent / "plugin" / "plugin.yaml").read_text(encoding="utf-8")


def _list(key):
    m = re.search(rf"^{key}:\n((?:  - .+\n)+)", MANIFEST, re.M)
    return sorted(x.strip()[2:] for x in m.group(1).splitlines()) if m else []


def test_declared_capabilities_match_registrations():
    ctx = FakeCtx()
    tironian.register(ctx)
    assert sorted(ctx.hooks) == _list("provides_hooks")
    assert sorted(ctx.middleware) == _list("provides_middleware") == ["llm_execution"]
    assert "provides_tools: []" in MANIFEST
    assert ctx.commands.keys() == {"tironian"} and ctx.cli.keys() == {"tironian"}
    assert ctx.aux == ["tironian_judge"]


def test_config_schema_matches_settings_defaults():
    schema_keys = set(re.findall(r"^  ([a-z_]+): \{", MANIFEST, re.M))
    settings_keys = {f.name for f in fields(Settings)}
    assert settings_keys <= schema_keys
    assert schema_keys - settings_keys == {"modal_token_id", "modal_token_secret"}


def test_register_does_not_open_database(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tironian.register(FakeCtx())
    assert not (tmp_path / "plugin-data").exists()


def test_slash_command_answers(tmp_path, monkeypatch):
    from tironian.core import app as app_mod
    app_mod.reset_app()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ctx = FakeCtx()
    tironian.register(ctx)
    assert "Learning from 0 turns" in ctx.commands["tironian"]("", session_id="s", platform="cli")
    assert "Tironian:" in ctx.commands["tironian"]("help")
    app_mod.reset_app()
