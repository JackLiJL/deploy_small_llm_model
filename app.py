import json
import httpx
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from models.registry import registry, ollama_client, ModelMetadata

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()

app = FastAPI(title="LLM API Proxy", version="1.0.0")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Configuration - can be overridden by environment variables
import os
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3.5:4b")


class ChatRequest(BaseModel):
    prompt: str
    model: str = DEFAULT_MODEL

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v):
        if not v.strip():
            raise ValueError("prompt must not be empty")
        return v.strip()


class ModelInfo(BaseModel):
    model_id: str
    version: str
    name: str
    status: str
    created_at: str
    ollama_details: dict = {}
    ollama_parameters: str = ""
    ollama_template: str = ""


class RegisterModelRequest(BaseModel):
    model_name: str
    version: str = ""
    status: str = "registered"


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning("rate_limit_exceeded", client=request.client.host if request.client else "unknown")
    return JSONResponse(status_code=429, content={"error": f"Rate limit exceeded: {exc.detail}"})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=500, content={"error": "internal server error"})


@app.get("/health")
@limiter.exempt
async def health(request: Request):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("http://host.docker.internal:11434/api/tags", timeout=3)
            r.raise_for_status()
        logger.info("health_check", status="healthy")
        return {"status": "healthy", "ollama": "connected"}
    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        return {"status": "degraded", "ollama": "unreachable"}


def _model_to_info(m: ModelMetadata) -> dict:
    return ModelInfo(
        model_id=m.model_id,
        version=m.version,
        name=m.name,
        status=m.status,
        created_at=m.created_at.isoformat(),
        ollama_details=m.ollama_details,
        ollama_parameters=m.ollama_parameters,
        ollama_template=m.ollama_template,
    ).model_dump()


@app.get("/models")
@limiter.exempt
async def list_models(request: Request):
    """List available models from the registry."""
    models = registry.list_models()
    return {"models": [_model_to_info(m) for m in models]}


@app.get("/models/{model_id}")
@limiter.exempt
async def get_model(model_id: str, request: Request):
    """Get specific model metadata."""
    models = registry.list_models(model_id)
    if not models:
        return JSONResponse(status_code=404, content={"error": f"Model {model_id} not found"})
    return {"models": [_model_to_info(m) for m in models]}


@app.post("/models/register")
@limiter.exempt
async def register_model(request: Request, req: RegisterModelRequest):
    """Register a model by fetching its info from Ollama."""
    try:
        version = req.version or None
        key = await registry.register_model_from_ollama(req.model_name, version, req.status)
        idx = key.rfind(":")
        model_id = key[:idx]
        ver = key[idx+1:]
        metadata = registry.get_model_metadata(model_id, ver)
        return {"message": "Model registered", "key": key, "model": _model_to_info(metadata)}
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        logger.error("register_model_failed", error=str(e))
        return JSONResponse(status_code=500, content={"error": "failed to register model"})


@app.on_event("startup")
async def startup_event():
    """Initialize default models on startup."""
    # Register default model if not exists
    default_model = registry.get_production_model(DEFAULT_MODEL)
    if not default_model:
        metadata = ModelMetadata(
            model_id=DEFAULT_MODEL,
            version="1.0.0",
            name="Qwen 3.5 4B",
            description="Default thinking model",
            tags=["default", "thinking"],
            status="production"
        )
        registry.register_model(metadata)
        logger.info("default_model_registered", model_id=DEFAULT_MODEL)


async def ollama_stream(prompt: str, model: str):
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                OLLAMA_URL,
                json={"model": model, "prompt": prompt, "stream": True, "think": True},
            ) as response:
                response.raise_for_status()
                thinking_done = False
                response_started = False

                async for line in response.aiter_lines():
                    if not line or not line.strip():
                        continue
                    try:
                        data = json.loads(line)

                        thinking_text = data.get("thinking")
                        if thinking_text:
                            if not thinking_done:
                                yield b"\033[90m[THINKING]\n"
                                thinking_done = True
                            yield thinking_text.encode("utf-8")

                        response_text = data.get("response")
                        if response_text:
                            if not response_started:
                                yield b"\033[0m\n[RESPONSE]\n"
                                response_started = True
                            yield response_text.encode("utf-8")
                    except Exception:
                        continue

                yield b"\033[0m\n"
    except httpx.ConnectError:
        logger.error("ollama_connection_failed", error="connection refused")
        yield b"Error: unable to connect to Ollama\n"
    except httpx.HTTPStatusError as e:
        logger.error("ollama_http_error", status=e.response.status_code)
        yield b"Error: Ollama returned an error\n"
    except Exception as e:
        logger.error("ollama_stream_error", error=str(e))
        yield b"Error: unexpected error\n"


@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, req: ChatRequest):
    logger.info("chat_request", prompt_length=len(req.prompt), model=req.model)
    return StreamingResponse(ollama_stream(req.prompt, req.model), media_type="text/plain")
