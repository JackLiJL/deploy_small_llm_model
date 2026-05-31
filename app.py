import requests
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

# Replace with your Ollama API URL if different
OLLAMA_URL = "http://host.docker.internal:11434/api/generate"

def ollama_stream(prompt: str):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "qwen3.5:4b",
                "prompt": prompt,
                "stream": True
            },
            stream=True
        )
        response.raise_for_status()
    except Exception:
        yield b"Error: unable to connect to Ollama\n"
        return

    thinking_done = False
    response_started = False
    
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.strip():
            continue
        try:
            data = json.loads(line)
            
            # Stream thinking tokens
            thinking_text = data.get("thinking")
            if thinking_text:
                if not thinking_done:
                    yield b"\033[90m[THINKING]\n"  # Dark grey
                yield thinking_text.encode("utf-8")
            
            # Stream response tokens
            response_text = data.get("response")
            if response_text:
                if not response_started:
                    if thinking_text or not thinking_done:  # Add separator if we had thinking
                        yield b"\033[0m\n[RESPONSE]\n"  # Reset color
                    response_started = True
                    thinking_done = True
                yield response_text.encode("utf-8")
        except Exception:
            continue
    
    yield b"\033[0m\n"  # Reset color and newline at end


@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        ollama_stream(req.prompt),
        media_type="text/plain"
    )
