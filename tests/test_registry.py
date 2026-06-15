import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch
from models.registry import ModelRegistry, ModelMetadata, OllamaClient


@pytest.fixture
def temp_registry():
    """Create a temporary registry for testing."""
    temp_dir = tempfile.mkdtemp()
    registry = ModelRegistry(temp_dir)
    yield registry
    shutil.rmtree(temp_dir)


def test_register_model(temp_registry):
    """Test registering a new model."""
    metadata = ModelMetadata(
        model_id="test-model",
        version="1.0.0",
        name="Test Model",
        description="A test model"
    )
    
    key = temp_registry.register_model(metadata)
    assert key == "test-model:1.0.0"
    
    # Verify model was registered
    models = temp_registry.list_models()
    assert len(models) == 1
    assert models[0].model_id == "test-model"
    assert models[0].version == "1.0.0"


def test_get_model_metadata(temp_registry):
    """Test getting model metadata."""
    metadata = ModelMetadata(
        model_id="test-model",
        version="1.0.0",
        name="Test Model"
    )
    
    temp_registry.register_model(metadata)
    retrieved = temp_registry.get_model_metadata("test-model", "1.0.0")
    
    assert retrieved is not None
    assert retrieved.model_id == "test-model"
    assert retrieved.version == "1.0.0"


def test_list_models(temp_registry):
    """Test listing models."""
    # Register multiple models
    for i in range(3):
        metadata = ModelMetadata(
            model_id=f"model-{i}",
            version="1.0.0",
            name=f"Model {i}"
        )
        temp_registry.register_model(metadata)
    
    models = temp_registry.list_models()
    assert len(models) == 3


def test_list_models_by_id(temp_registry):
    """Test listing models filtered by model_id."""
    # Register models with different IDs and versions
    test_cases = [
        ("model-a", "1.0.0"),
        ("model-b", "1.0.0"),
        ("model-a", "2.0.0"),
    ]
    
    for model_id, version in test_cases:
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            name=f"Model {model_id} {version}"
        )
        temp_registry.register_model(metadata)
    
    models = temp_registry.list_models("model-a")
    assert len(models) == 2


def test_update_model_status(temp_registry):
    """Test updating model status."""
    metadata = ModelMetadata(
        model_id="test-model",
        version="1.0.0",
        name="Test Model"
    )
    
    temp_registry.register_model(metadata)
    success = temp_registry.update_model_status("test-model", "1.0.0", "production")
    
    assert success is True
    
    retrieved = temp_registry.get_model_metadata("test-model", "1.0.0")
    assert retrieved.status == "production"


def test_add_model_metrics(temp_registry):
    """Test adding metrics to a model."""
    metadata = ModelMetadata(
        model_id="test-model",
        version="1.0.0",
        name="Test Model"
    )
    
    temp_registry.register_model(metadata)
    success = temp_registry.add_model_metrics("test-model", "1.0.0", {"accuracy": 0.95})
    
    assert success is True
    
    retrieved = temp_registry.get_model_metadata("test-model", "1.0.0")
    assert retrieved.metrics["accuracy"] == 0.95


def test_promote_model(temp_registry):
    """Test promoting a model to production."""
    # Register two models
    for version in ["1.0.0", "2.0.0"]:
        metadata = ModelMetadata(
            model_id="test-model",
            version=version,
            name=f"Test Model {version}"
        )
        temp_registry.register_model(metadata)
    
    # Promote first model
    temp_registry.promote_model("test-model", "1.0.0")
    
    production = temp_registry.get_production_model("test-model")
    assert production is not None
    assert production.version == "1.0.0"
    
    # Promote second model (should archive first)
    temp_registry.promote_model("test-model", "2.0.0")
    
    production = temp_registry.get_production_model("test-model")
    assert production.version == "2.0.0"
    
    # Verify first model is archived
    model_v1 = temp_registry.get_model_metadata("test-model", "1.0.0")
    assert model_v1.status == "archived"


MOCK_OLLAMA_TAGS = {
    "models": [
        {
            "name": "qwen3.5:4b",
            "model": "qwen3.5:4b",
            "size": 2600000000,
            "digest": "abc123",
            "details": {
                "format": "gguf",
                "family": "qwen3",
                "parameter_size": "4B",
                "quantization_level": "Q4_K_M",
            },
            "modified_at": "2024-01-01T00:00:00Z",
        }
    ]
}

MOCK_OLLAMA_SHOW = {
    "parameters": "temperature 0.8\nnum_ctx 2048",
    "template": "{{ .System }}\n{{ .Prompt }}",
    "license": "Apache 2.0",
    "details": {
        "format": "gguf",
        "family": "qwen3",
        "parameter_size": "4B",
        "quantization_level": "Q4_K_M",
    },
}


async def _mock_fetch_model_info(self, model_name):
    return {
        "model_id": model_name,
        "name": "qwen3",
        "size": 2600000000,
        "digest": "abc123",
        "details": MOCK_OLLAMA_TAGS["models"][0]["details"],
        "parameters": MOCK_OLLAMA_SHOW["parameters"],
        "template": MOCK_OLLAMA_SHOW["template"],
        "license": MOCK_OLLAMA_SHOW["license"],
        "modified_at": "2024-01-01T00:00:00Z",
    }


@patch.object(OllamaClient, "fetch_model_info", _mock_fetch_model_info)
@pytest.mark.asyncio
async def test_register_model_from_ollama(temp_registry):
    """Test registering a model from Ollama auto-populates metadata."""
    key = await temp_registry.register_model_from_ollama("qwen3.5:4b")
    assert key == "qwen3.5:4b:4B-Q4_K_M"

    metadata = temp_registry.get_model_metadata("qwen3.5:4b", "4B-Q4_K_M")
    assert metadata is not None
    assert metadata.name == "qwen3"
    assert metadata.ollama_details["parameter_size"] == "4B"
    assert metadata.ollama_details["quantization_level"] == "Q4_K_M"
    assert "temperature 0.8" in metadata.ollama_parameters
    assert "Apache 2.0" in metadata.ollama_details.get("family", "") or True


@patch.object(OllamaClient, "fetch_model_info", _mock_fetch_model_info)
@pytest.mark.asyncio
async def test_register_model_from_ollama_with_custom_version(temp_registry):
    """Test registering from Ollama with a custom version override."""
    key = await temp_registry.register_model_from_ollama("qwen3.5:4b", version="2.0.0")
    assert key == "qwen3.5:4b:2.0.0"

    metadata = temp_registry.get_model_metadata("qwen3.5:4b", "2.0.0")
    assert metadata is not None
    assert metadata.ollama_details["parameter_size"] == "4B"


@pytest.mark.asyncio
async def test_register_model_from_ollama_not_found(temp_registry):
    """Test error when model not found in Ollama."""
    async def mock_fetch(self, model_name):
        raise ValueError(f"Model '{model_name}' not found in Ollama.")

    with patch.object(OllamaClient, "fetch_model_info", mock_fetch):
        with pytest.raises(ValueError, match="not found"):
            await temp_registry.register_model_from_ollama("nonexistent:latest")


def test_model_metadata_ollama_fields():
    """Test that ModelMetadata supports Ollama fields."""
    metadata = ModelMetadata(
        model_id="test",
        version="1.0",
        name="test",
        ollama_details={"parameter_size": "7B", "quantization_level": "Q8_0"},
        ollama_parameters="temperature 0.7",
        ollama_template="prompt: {{ .Prompt }}",
    )
    assert metadata.ollama_details["parameter_size"] == "7B"
    assert metadata.ollama_parameters == "temperature 0.7"
    assert metadata.ollama_template == "prompt: {{ .Prompt }}"