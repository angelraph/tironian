"""The trainer's processes see an allow-list of Hermes's environment, never its secrets."""

from plugin.core import jobs

SECRETS = {
    "OPENAI_API_KEY": "sk-1", "ANTHROPIC_API_KEY": "sk-2", "OPENROUTER_API_KEY": "sk-3",
    "NOUS_API_KEY": "n-1", "SLACK_BOT_TOKEN": "xoxb-1", "TELEGRAM_BOT_TOKEN": "t-1", "DISCORD_TOKEN": "d-1",
    "GITHUB_TOKEN": "ghp-1", "AWS_SECRET_ACCESS_KEY": "a-1", "HF_TOKEN": "hf-1",
    "MODAL_TOKEN_ID": "from-env", "PYTHONPATH": "/hermes/site-packages", "PIP_INDEX_URL": "https://x",
}


def _patch_env(monkeypatch):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")


def test_only_allowed_names_reach_the_trainer(monkeypatch):
    _patch_env(monkeypatch)
    env = jobs.clean_env({"MODAL_TOKEN_ID": "id", "MODAL_TOKEN_SECRET": "secret"})
    for k in SECRETS:
        if k != "MODAL_TOKEN_ID":
            assert k not in env, k
    assert env["MODAL_TOKEN_ID"] == "id"            # from Hermes's secret store, not the environment
    assert env["MODAL_TOKEN_SECRET"] == "secret"
    assert env["PATH"] == "/usr/bin" and env["LC_ALL"] == "C.UTF-8" and env["HTTPS_PROXY"]


def test_hugging_face_and_cuda_only_for_local_training(monkeypatch):
    _patch_env(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert "HF_TOKEN" not in jobs.clean_env() and "CUDA_VISIBLE_DEVICES" not in jobs.clean_env()
    local = jobs.clean_env(local_training=True)
    assert local["HF_TOKEN"] == "hf-1" and local["CUDA_VISIBLE_DEVICES"] == "0"
    assert "SLACK_BOT_TOKEN" not in local and "OPENAI_API_KEY" not in local


def test_only_the_modal_token_is_forwarded_from_secrets():
    assert jobs._modal_secrets({"MODAL_TOKEN_ID": "a", "MODAL_TOKEN_SECRET": "b", "OPENAI_API_KEY": "c"}) == \
        {"MODAL_TOKEN_ID": "a", "MODAL_TOKEN_SECRET": "b"}


def test_the_detached_trainer_gets_the_allow_list(monkeypatch, tmp_path):
    _patch_env(monkeypatch)
    seen = {}

    class FakePopen:
        def __init__(self, cmd, **kw):
            seen.update(kw["env"])
            self.pid = 4242
    monkeypatch.setattr(jobs.subprocess, "Popen", FakePopen)
    jobs._spawn(tmp_path, tmp_path, {"MODAL_TOKEN_ID": "id", "MODAL_TOKEN_SECRET": "s", "GITHUB_TOKEN": "g"},
                dry=False)
    assert seen["MODAL_TOKEN_ID"] == "id" and "GITHUB_TOKEN" not in seen and "SLACK_BOT_TOKEN" not in seen
    seen.clear()
    jobs._spawn(tmp_path, tmp_path, {"MODAL_TOKEN_ID": "id", "MODAL_TOKEN_SECRET": "s"}, dry=False, local=True)
    assert "MODAL_TOKEN_ID" not in seen               # a local run never talks to Modal
