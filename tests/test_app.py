import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from httpx import ConnectError, HTTPStatusError, Response

from app import app, ChatRequest

client = TestClient(app)


def test_health_healthy():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch("app.httpx.AsyncClient") as mock_cls:
        instance = AsyncMock()
        instance.get = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = instance

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["ollama"] == "connected"


def test_health_degraded():
    with patch("app.httpx.AsyncClient") as mock_cls:
        instance = AsyncMock()
        instance.get = AsyncMock(side_effect=ConnectError("refused"))
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = instance

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["ollama"] == "unreachable"


def test_chat_validation_empty_prompt():
    resp = client.post("/chat", json={"prompt": ""})
    assert resp.status_code == 422


def test_chat_validation_whitespace_only():
    resp = client.post("/chat", json={"prompt": "   "})
    assert resp.status_code == 422


def test_chat_validation_missing_prompt():
    resp = client.post("/chat", json={})
    assert resp.status_code == 422


def test_chat_request_strips_whitespace():
    req = ChatRequest(prompt="  hello  ")
    assert req.prompt == "hello"


def test_chat_request_default_model():
    req = ChatRequest(prompt="hello")
    assert req.model == "qwen3.5:4b"


def test_chat_connection_error():
    with patch("app.ollama_stream") as mock_stream:
        async def gen():
            yield b"Error: unable to connect to Ollama\n"

        mock_stream.return_value = gen()

        resp = client.post("/chat", json={"prompt": "hello"})
        assert resp.status_code == 200
        assert b"Error: unable to connect to Ollama" in resp.content


def test_chat_streaming_response():
    with patch("app.ollama_stream") as mock_stream:
        async def gen():
            yield b"\033[90m[THINKING]\n"
            yield b"thinking..."
            yield b"\033[0m\n[RESPONSE]\n"
            yield b"Hello!"
            yield b"\033[0m\n"

        mock_stream.return_value = gen()

        resp = client.post("/chat", json={"prompt": "hello"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/plain; charset=utf-8"


def test_chat_model_override():
    with patch("app.ollama_stream") as mock_stream:
        async def gen():
            yield b"response"

        mock_stream.return_value = gen()

        resp = client.post("/chat", json={"prompt": "hello", "model": "custom:1b"})
        assert resp.status_code == 200
        mock_stream.assert_called_once_with("hello", "custom:1b")


def test_rate_limit_returns_429():
    with patch("app.ollama_stream") as mock_stream:
        async def gen():
            yield b"ok"

        mock_stream.return_value = gen()

        for _ in range(10):
            client.post("/chat", json={"prompt": "test"})

        resp = client.post("/chat", json={"prompt": "test"})
        assert resp.status_code == 429
        data = resp.json()
        assert "Rate limit exceeded" in data["error"]
