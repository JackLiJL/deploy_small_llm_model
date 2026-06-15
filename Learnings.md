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

## Day 3 — MLOps: Model Versioning & Registry

- **Model Registry Pattern** — A model registry tracks which model versions are available, their metadata, and their deployment status (registered → staging → production → archived).

- **File-based Registry** — Simple file-based registry using JSON files for metadata. Good for learning; production systems use databases (MLflow, DVC).

- **Model Metadata** — Important metadata includes: model_id, version, name, description, created_at, tags, metrics, parameters, status.

- **Production Promotion** — When promoting a new model to production, archive the previous production model. This enables rollback by promoting the archived version.

- **Ollama API for Model Metadata** — `GET /api/tags` returns model list with size, digest, details (format, family, parameter_size, quantization_level). `POST /api/show` returns parameters, template, license. Both are needed for complete model metadata.

- **Docker Networking Pitfall** — Inside a Docker container, `localhost` refers to the container itself, not the host machine. Use `host.docker.internal` (with `--add-host=host.docker.internal:host-gateway`) to reach services on the host. The OllamaClient default URL must match this.
