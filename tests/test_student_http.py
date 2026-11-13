"""Calls to the local student: never through a proxy, never an unbounded read."""

from __future__ import annotations

from tironian.core import student
from tironian.core.student import StudentClient, _is_loopback


def test_local_calls_skip_a_configured_proxy(llama, monkeypatch):
    # A judge's tests failed because HTTP_PROXY sent the 127.0.0.1 call to a proxy.
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")   # nothing listens on port 9
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    client = StudentClient(llama.url)
    assert client.available_models() == [llama.model]
    reply = client.complete(llama.model, [{"role": "user", "content": "hi"}])
    assert reply.ok and reply.text.endswith("ok")


def test_an_oversized_reply_falls_back(llama, monkeypatch):
    monkeypatch.setattr(student, "MAX_REPLY_BYTES", 200)
    llama.replies = ["x" * 1000]
    reply = StudentClient(llama.url).complete(llama.model, [{"role": "user", "content": "hi"}])
    assert not reply.ok and reply.error == "too_large"


def test_loopback_detection():
    assert _is_loopback("http://127.0.0.1:11434/v1")
    assert _is_loopback("http://localhost:8080/v1/models")
    assert _is_loopback("http://[::1]:8080/v1")
    assert _is_loopback("http://ollama.localhost/v1")
    assert not _is_loopback("http://192.168.1.20:11434/v1")
    assert not _is_loopback("https://example.com/v1")
