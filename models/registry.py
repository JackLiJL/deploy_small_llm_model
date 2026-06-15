import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import httpx
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()


class ModelMetadata(BaseModel):
    """Metadata for a registered model version."""
    model_id: str
    version: str
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    tags: list[str] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    parameters: dict = Field(default_factory=dict)
    status: str = "registered"  # registered, staging, production, archived
    ollama_details: dict = Field(default_factory=dict)
    ollama_parameters: str = ""
    ollama_template: str = ""


class OllamaClient:
    """Fetch model information from Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    async def list_available_models(self) -> list[dict]:
        """List all locally available Ollama models."""
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            return r.json().get("models", [])

    async def fetch_model_info(self, model_name: str) -> dict:
        """Fetch detailed info for a specific model from Ollama."""
        models = await self.list_available_models()
        matched = next((m for m in models if m["name"] == model_name), None)
        if not matched:
            raise ValueError(f"Model '{model_name}' not found in Ollama. Available: {[m['name'] for m in models]}")

        details = matched.get("details", {})
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=5,
            )
            r.raise_for_status()
            show = r.json()

        return {
            "model_id": model_name,
            "name": show.get("details", {}).get("family", model_name),
            "size": matched.get("size", 0),
            "digest": matched.get("digest", ""),
            "details": details,
            "parameters": show.get("parameters", ""),
            "template": show.get("template", ""),
            "license": show.get("license", ""),
            "modified_at": matched.get("modified_at", ""),
        }


class ModelRegistry:
    """Simple file-based model registry for tracking model versions."""
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._ensure_metadata_file()
    
    def _ensure_metadata_file(self):
        """Ensure the metadata file exists."""
        metadata_file = self.registry_path / "metadata.json"
        if not metadata_file.exists():
            metadata_file.write_text("{}")
    
    def _get_model_path(self, model_id: str, version: str) -> Path:
        """Get the path for a specific model version."""
        return self.registry_path / model_id / version
    
    def _load_metadata(self) -> dict:
        """Load metadata from file."""
        metadata_file = self.registry_path / "metadata.json"
        return json.loads(metadata_file.read_text())
    
    def _save_metadata(self, metadata: dict):
        """Save metadata to file."""
        metadata_file = self.registry_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2, default=str))
    
    def register_model(self, metadata: ModelMetadata) -> str:
        """Register a new model version."""
        model_path = self._get_model_path(metadata.model_id, metadata.version)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model metadata
        metadata_file = model_path / "metadata.json"
        metadata_file.write_text(metadata.model_dump_json(indent=2))
        
        # Update global metadata
        all_metadata = self._load_metadata()
        key = f"{metadata.model_id}:{metadata.version}"
        all_metadata[key] = metadata.model_dump()
        self._save_metadata(all_metadata)
        
        logger.info("model_registered", model_id=metadata.model_id, version=metadata.version)
        return key
    
    def get_model_metadata(self, model_id: str, version: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model version."""
        model_path = self._get_model_path(model_id, version)
        metadata_file = model_path / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        return ModelMetadata.model_validate_json(metadata_file.read_text())
    
    def list_models(self, model_id: Optional[str] = None) -> list[ModelMetadata]:
        """List all registered models, optionally filtered by model_id."""
        all_metadata = self._load_metadata()
        models = []
        
        for key, data in all_metadata.items():
            if model_id and not key.startswith(model_id):
                continue
            models.append(ModelMetadata(**data))
        
        return sorted(models, key=lambda x: x.created_at, reverse=True)
    
    def update_model_status(self, model_id: str, version: str, status: str) -> bool:
        """Update the status of a model version."""
        metadata = self.get_model_metadata(model_id, version)
        if not metadata:
            return False
        
        metadata.status = status
        metadata.updated_at = datetime.now()
        
        # Save updated metadata
        model_path = self._get_model_path(model_id, version)
        metadata_file = model_path / "metadata.json"
        metadata_file.write_text(metadata.model_dump_json(indent=2))
        
        # Update global metadata
        all_metadata = self._load_metadata()
        key = f"{model_id}:{version}"
        all_metadata[key] = metadata.model_dump()
        self._save_metadata(all_metadata)
        
        logger.info("model_status_updated", model_id=model_id, version=version, status=status)
        return True
    
    def add_model_metrics(self, model_id: str, version: str, metrics: dict) -> bool:
        """Add or update metrics for a model version."""
        metadata = self.get_model_metadata(model_id, version)
        if not metadata:
            return False
        
        metadata.metrics.update(metrics)
        metadata.updated_at = datetime.now()
        
        # Save updated metadata
        model_path = self._get_model_path(model_id, version)
        metadata_file = model_path / "metadata.json"
        metadata_file.write_text(metadata.model_dump_json(indent=2))
        
        # Update global metadata
        all_metadata = self._load_metadata()
        key = f"{model_id}:{version}"
        all_metadata[key] = metadata.model_dump()
        self._save_metadata(all_metadata)
        
        logger.info("model_metrics_updated", model_id=model_id, version=version, metrics=metrics)
        return True
    
    def get_production_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get the current production model for a given model_id."""
        models = self.list_models(model_id)
        for model in models:
            if model.status == "production":
                return model
        return None
    
    def promote_model(self, model_id: str, version: str) -> bool:
        """Promote a model to production status."""
        # First, archive any existing production model
        current_production = self.get_production_model(model_id)
        if current_production:
            self.update_model_status(model_id, current_production.version, "archived")
        
        # Promote the new model
        return self.update_model_status(model_id, version, "production")

    async def register_model_from_ollama(
        self, model_name: str, version: str = None, status: str = "registered"
    ) -> str:
        """Register a model by fetching its info from Ollama."""
        info = await ollama_client.fetch_model_info(model_name)

        details = info.get("details", {})
        if not version:
            param_size = details.get("parameter_size", "unknown")
            quant = details.get("quantization_level", "unknown")
            version = f"{param_size}-{quant}"

        metadata = ModelMetadata(
            model_id=model_name,
            version=version,
            name=info.get("name", model_name),
            description=f"Auto-registered from Ollama: {details.get('family', '')} {details.get('parameter_size', '')} {details.get('quantization_level', '')}".strip(),
            tags=[details.get("family", ""), details.get("quantization_level", "")],
            status=status,
            ollama_details=details,
            ollama_parameters=info.get("parameters", ""),
            ollama_template=info.get("template", ""),
        )

        return self.register_model(metadata)


# Global instances
ollama_client = OllamaClient(
    os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
)
registry = ModelRegistry()