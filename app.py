import json
import httpx
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

OLLAMA_URL = "http://host.docker.internal:11434/api/generate"


class ChatRequest(BaseModel):
    prompt: str
    model: str = "qwen3.5:4b"

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v):
        if not v.strip():
            raise ValueError("prompt must not be empty")
        return v.strip()


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
