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
