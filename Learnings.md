# Learnings

## Day 1 — LLM Serving Fundamentals

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

## Day 2 — Thinking Models & Production Hardening

- **Thinking model API quirks** — Thinking models (like qwen3.5) require `"think": true` in the Ollama request. Without it, thinking tokens leak into the `response` field as raw `[THINKING]` text instead of being separated into the `thinking` field.

- **Docker port mapping** — `docker run` without `-p` exposes ports only inside the container. Always use `-p 8000:8000` to map host ↔ container.

- **Volume mounts vs rebuilds** — `-v $(pwd)/app.py:/app/app.py` syncs code live (no rebuild needed). But new pip dependencies still require `docker build`.

- **stdout is the Docker-native logging path** — Structured JSON to stdout gets captured by `docker logs`. No file-based logging needed in containers.

- **Async matters** — Synchronous `requests` inside an async FastAPI endpoint blocks the event loop. `httpx.AsyncClient` fixes this.

- **Rate limiting with slowapi** — The decorator requires a `request` parameter on the endpoint function, even if you don't use it.
