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

## Study Notes

### Day 1 — LLM Serving Fundamentals

- **LLM serving is a distributed systems problem**
  - 3 separate systems: EC2 host, Ollama runtime (model server), FastAPI app (docker container)
  - Request path:
    ```
    curl → FastAPI container → HTTP → Ollama service → model inference
    ```

- **Infrastructure is critical** - Infrastructure breaks before code does in ML systems
  - Memory: Model requires 1.3 GiB; available 432 GiB
  - Disk: Not enough volume
  
- **Connection debugging** - When Ollama runs but API fails with "connection refused":
  - Process might be running but not listening correctly
  - Crashed after startup
  - Bound to wrong interface
  - Out of memory mid-request

- **Memory is part of model correctness** - In LLMs, memory determines whether the model exists at all

### Day 2 — Thinking Models & Production Hardening

- **Thinking model API quirks** — Thinking models (like qwen3.5) require `"think": true` in the Ollama request. Without it, thinking tokens leak into the `response` field as raw `[THINKING]` text instead of being separated into the `thinking` field.

- **Docker port mapping** — `docker run` without `-p` exposes ports only inside the container. Always use `-p 8000:8000` to map host ↔ container.

- **Volume mounts vs rebuilds** — `-v $(pwd)/app.py:/app/app.py` syncs code live (no rebuild needed). But new pip dependencies still require `docker build`.

- **stdout is the Docker-native logging path** — Structured JSON to stdout gets captured by `docker logs`. No file-based logging needed in containers.

- **Async matters** — Synchronous `requests` inside an async FastAPI endpoint blocks the event loop. `httpx.AsyncClient` fixes this.

- **Rate limiting with slowapi** — The decorator requires a `request` parameter on the endpoint function, even if you don't use it.
