"""Tests for triage.config. Model calls are mocked: tests never hit an API."""

from unittest.mock import patch

from triage import config


def setup_function():
    # get_chat_model/get_embeddings are cached; clear so each test starts fresh.
    config.get_chat_model.cache_clear()
    config.get_embeddings.cache_clear()


def test_chat_model_reads_env_and_uses_rate_limiter(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "groq:some-model")
    with patch.object(config, "init_chat_model") as fake_init:
        config.get_chat_model()

    args, kwargs = fake_init.call_args
    assert args[0] == "groq:some-model"
    assert kwargs["rate_limiter"] is config._rate_limiter
    assert kwargs["max_retries"] >= 1


def test_chat_model_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    with patch.object(config, "init_chat_model") as fake_init:
        config.get_chat_model()
    assert fake_init.call_args.args[0] == config.DEFAULT_CHAT_MODEL


def test_rate_limit_is_conservative():
    # Free tier allows only a few requests per minute; stay at <= 6 RPM.
    assert config._rate_limiter.requests_per_second <= 0.1


def test_embeddings_use_env_model_on_cpu(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "some/embedder")
    with patch.object(config, "HuggingFaceEmbeddings") as fake_emb:
        config.get_embeddings()
    kwargs = fake_emb.call_args.kwargs
    assert kwargs["model_name"] == "some/embedder"
    assert kwargs["model_kwargs"]["device"] == "cpu"
