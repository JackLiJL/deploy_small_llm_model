import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Request(BaseModel):
    prompt: str

# Replace with your actual Ollama API URL
OLLAMA_URL = "http://localhost:11434/api/generate"

@app.post("/chat")
def chat(req: Request):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "llama3.2:1b",
            "prompt": req.prompt,
            "stream": False
        }
    )

    return response.json()
