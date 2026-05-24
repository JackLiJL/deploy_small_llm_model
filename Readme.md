# A Simple LLM API Project

## Overview

LLM serving is a distributed systems problem. This project provides a simple API for deploying and serving small language models.

## Features

- Fast LLM inference API
- Docker containerization
- Easy deployment

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (optional)

### Installation

```bash
pip install -r requirements.txt
```

### Running Locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
docker build -t llm-api .
docker run -p 8000:8000 llm-api
```

## My Study Notes

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
