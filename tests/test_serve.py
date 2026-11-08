"""Loading a student for the user: Ollama Modelfile, endpoint choice, model names."""

from tironian.core import serve
from tironian.core.student import model_matches


def test_modelfile_matches_the_training_prompt():
    mf = serve.MODELFILE.format(gguf="tironian-f1-v1.gguf")
    assert mf.startswith("FROM ./tironian-f1-v1.gguf\n")
    # Same opening as train_core.GEN_PREFIX: an empty, closed think block before the reply.
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n\"\"\"" in mf
    assert "{{ .Role }}" in mf and "{{ .Content }}" in mf and 'PARAMETER stop "<|im_end|>"' in mf


def test_ollama_names_match():
    assert model_matches("tironian-f1-v1:latest", "tironian-f1-v1")
    assert not model_matches("tironian-f1-v2:latest", "tironian-f1-v1")


def test_keep_warm_asks_ollama_to_keep_the_student_loaded():
    """Found live: Ollama unloaded the student after 5 idle minutes and the reload blew the deadline."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert serve.keep_warm(f"http://127.0.0.1:{srv.server_port}/v1", "tironian-f1-v1")
    finally:
        srv.shutdown()
    assert seen["path"] == "/api/generate"
    assert seen["body"]["model"] == "tironian-f1-v1" and seen["body"]["keep_alive"] == "60m"
    assert serve.is_ollama("http://127.0.0.1:11434/v1") and not serve.is_ollama("http://127.0.0.1:8080/v1")


def test_custom_endpoint_is_never_replaced():
    assert serve.resolve_endpoint("http://10.0.0.5:9000/v1") == "http://10.0.0.5:9000/v1"


def test_default_endpoint_falls_back_to_ollama(monkeypatch):
    serve.forget()
    monkeypatch.setattr(serve, "_lists", lambda endpoint, model_id: endpoint == serve.OLLAMA_ENDPOINT)
    assert serve.resolve_endpoint(serve.HERMES_ENDPOINT, "tironian-f1-v1") == serve.OLLAMA_ENDPOINT
    serve.forget()
    monkeypatch.setattr(serve, "_lists", lambda endpoint, model_id: True)
    assert serve.resolve_endpoint(serve.HERMES_ENDPOINT, "tironian-f1-v1") == serve.HERMES_ENDPOINT
    serve.forget()


def test_load_without_ollama_explains(monkeypatch, tmp_path):
    monkeypatch.setattr(serve, "find_ollama", lambda: None)
    ok, msg = serve.load_into_ollama(tmp_path / "x.gguf", "x")
    assert not ok and "Ollama" in msg
