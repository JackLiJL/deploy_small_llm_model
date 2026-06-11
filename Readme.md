# LLM API Proxy

A FastAPI service that proxies requests to a local Ollama instance, streaming responses with thinking token support.

## Features

- **Streaming responses** — streams thinking and response tokens from Ollama via SSE
- **Health check** — `/health` endpoint verifies Ollama connectivity
- **Rate limiting** — 10 requests/minute per client via slowapi
- **Request validation** — rejects empty prompts with structured error responses
- **Structured logging** — JSON logs to stdout (captured by `docker logs`)
- **Async HTTP** — uses httpx instead of blocking requests

## Quick Start

```bash
# With Docker (recommended)
docker run -d -p 8000:8000 -v "$(pwd)/app.py:/app/app.py" --name llm-api llm-api

# Or without Docker
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Usage

```bash
# Chat
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello"}'

# Health check
curl http://127.0.0.1:8000/health
```

## Development

```bash
# Run tests
pip install pytest httpx
pytest tests/

# Rebuild container after dependency changes
docker build -t llm-api . && docker run -d -p 8000:8000 -v "$(pwd)/app.py:/app/app.py" --name llm-api llm-api

# View logs
docker logs -f llm-api
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://host.docker.internal:11434/api/generate` | Ollama API endpoint |
| Model | `qwen3.5:4b` | Can be overridden per request via `model` field |
| Rate limit | `10/minute` | Per client IP |

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Stream chat response from Ollama |
| `GET` | `/health` | Check Ollama connectivity |
