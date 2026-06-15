# LLM API Proxy

A FastAPI service that proxies requests to a local Ollama instance, streaming responses with thinking token support, with model versioning via a file-based registry.

## Features

- **Streaming responses** — streams thinking and response tokens from Ollama via SSE
- **Model registry** — versioned model tracking with Ollama metadata auto-sync
- **Health check** — `/health` endpoint verifies Ollama connectivity
- **Rate limiting** — 10 requests/minute per client via slowapi
- **Request validation** — rejects empty prompts with structured error responses
- **Structured logging** — JSON logs to stdout (captured by `docker logs`)
- **Async HTTP** — uses httpx instead of blocking requests

## Quick Start

```bash
# Build and run with Docker
docker build -t llm-proxy .
docker run -d -p 8000:8000 --add-host=host.docker.internal:host-gateway --name llm-proxy llm-proxy

# Or without Docker
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Testing the Endpoints

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Sync a Model from Ollama
```bash
curl -X POST http://localhost:8000/models/register \
  -H "Content-Type: application/json" \
  -d '{"model_name": "qwen3.5:4b"}'
```

### 3. List Registered Models
```bash
curl http://localhost:8000/models
```

### 4. Get Model Details
```bash
curl http://localhost:8000/models/qwen3.5:4b
```

### 5. Chat with Model
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello"}'
```

## CLI (Model Registry)

```bash
# Sync a single model from Ollama
python -m models.cli sync --model-name qwen3.5:4b

# Sync all locally available Ollama models
python -m models.cli sync-all

# List registered models
python -m models.cli list

# Get model metadata
python -m models.cli get --model-id qwen3.5:4b --version 4.7B-Q4_K_M

# Update model status
python -m models.cli status --model-id qwen3.5:4b --version 4.7B-Q4_K_M --status production

# Add metrics
python -m models.cli metrics --model-id qwen3.5:4b --version 4.7B-Q4_K_M --metrics '{"accuracy": 0.95}'

# Promote to production
python -m models.cli promote --model-id qwen3.5:4b --version 4.7B-Q4_K_M
```

## Development

```bash
# Run tests
pip install pytest pytest-asyncio httpx
pytest tests/

# Rebuild container
docker build -t llm-proxy . && docker rm -f llm-proxy && \
docker run -d -p 8000:8000 --add-host=host.docker.internal:host-gateway --name llm-proxy llm-proxy

# View logs
docker logs -f llm-proxy
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://host.docker.internal:11434/api/generate` | Ollama API endpoint |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama base URL for registry sync |
| `DEFAULT_MODEL` | `qwen3.5:4b` | Default model for chat requests |
| Rate limit | `10/minute` | Per client IP |

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Stream chat response from Ollama |
| `GET` | `/health` | Check Ollama connectivity |
| `GET` | `/models` | List registered models with Ollama metadata |
| `GET` | `/models/{model_id}` | Get model details |
| `POST` | `/models/register` | Sync a model from Ollama into the registry |
