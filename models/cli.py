#!/usr/bin/env python3
"""CLI tool for managing the model registry."""

import argparse
import asyncio
import json
from datetime import datetime
from .registry import ModelRegistry, ModelMetadata, registry, ollama_client


def register_model(args):
    """Register a new model version."""
    metadata = ModelMetadata(
        model_id=args.model_id,
        version=args.version,
        name=args.name,
        description=args.description,
        tags=args.tags.split(",") if args.tags else [],
    )
    
    key = registry.register_model(metadata)
    print(f"Registered model: {key}")


def list_models(args):
    """List registered models."""
    models = registry.list_models(args.model_id)
    
    if not models:
        print("No models registered.")
        return
    
    print(f"{'Model ID':<20} {'Version':<10} {'Status':<12} {'Created':<20}")
    print("-" * 65)
    
    for model in models:
        print(f"{model.model_id:<20} {model.version:<10} {model.status:<12} {model.created_at.strftime('%Y-%m-%d %H:%M'):<20}")


def get_model(args):
    """Get model metadata."""
    metadata = registry.get_model_metadata(args.model_id, args.version)
    
    if not metadata:
        print(f"Model {args.model_id}:{args.version} not found.")
        return
    
    print(json.dumps(metadata.model_dump(), indent=2, default=str))


def update_status(args):
    """Update model status."""
    success = registry.update_model_status(args.model_id, args.version, args.status)
    
    if success:
        print(f"Updated {args.model_id}:{args.version} status to {args.status}")
    else:
        print(f"Failed to update model {args.model_id}:{args.version}")


def add_metrics(args):
    """Add metrics to a model."""
    try:
        metrics = json.loads(args.metrics)
    except json.JSONDecodeError:
        print("Invalid JSON for metrics.")
        return
    
    success = registry.add_model_metrics(args.model_id, args.version, metrics)
    
    if success:
        print(f"Added metrics to {args.model_id}:{args.version}")
    else:
        print(f"Failed to add metrics to {args.model_id}:{args.version}")


def promote_model(args):
    """Promote a model to production."""
    success = registry.promote_model(args.model_id, args.version)
    
    if success:
        print(f"Promoted {args.model_id}:{args.version} to production")
    else:
        print(f"Failed to promote model {args.model_id}:{args.version}")


def sync_model(args):
    """Sync a model from Ollama into the registry."""
    async def _sync():
        try:
            key = await registry.register_model_from_ollama(args.model_name, args.version, args.status)
            print(f"Synced model from Ollama: {key}")
        except ValueError as e:
            print(f"Error: {e}")
    asyncio.run(_sync())


def sync_all(args):
    """Sync all locally available Ollama models into the registry."""
    async def _sync_all():
        try:
            models = await ollama_client.list_available_models()
            if not models:
                print("No models found in Ollama.")
                return
            print(f"Found {len(models)} models in Ollama:")
            for model in models:
                name = model["name"]
                existing = registry.list_models(name)
                if existing:
                    print(f"  {name} - already registered, skipping")
                else:
                    try:
                        key = await registry.register_model_from_ollama(name)
                        print(f"  {name} - registered as {key}")
                    except Exception as e:
                        print(f"  {name} - failed: {e}")
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
    asyncio.run(_sync_all())


def main():
    parser = argparse.ArgumentParser(description="Model Registry CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Register model
    register_parser = subparsers.add_parser("register", help="Register a new model")
    register_parser.add_argument("--model-id", required=True, help="Model ID")
    register_parser.add_argument("--version", required=True, help="Model version")
    register_parser.add_argument("--name", required=True, help="Model name")
    register_parser.add_argument("--description", help="Model description")
    register_parser.add_argument("--tags", help="Comma-separated tags")
    register_parser.set_defaults(func=register_model)
    
    # List models
    list_parser = subparsers.add_parser("list", help="List models")
    list_parser.add_argument("--model-id", help="Filter by model ID")
    list_parser.set_defaults(func=list_models)
    
    # Get model
    get_parser = subparsers.add_parser("get", help="Get model metadata")
    get_parser.add_argument("--model-id", required=True, help="Model ID")
    get_parser.add_argument("--version", required=True, help="Model version")
    get_parser.set_defaults(func=get_model)
    
    # Update status
    status_parser = subparsers.add_parser("status", help="Update model status")
    status_parser.add_argument("--model-id", required=True, help="Model ID")
    status_parser.add_argument("--version", required=True, help="Model version")
    status_parser.add_argument("--status", required=True, choices=["registered", "staging", "production", "archived"])
    status_parser.set_defaults(func=update_status)
    
    # Add metrics
    metrics_parser = subparsers.add_parser("metrics", help="Add metrics to model")
    metrics_parser.add_argument("--model-id", required=True, help="Model ID")
    metrics_parser.add_argument("--version", required=True, help="Model version")
    metrics_parser.add_argument("--metrics", required=True, help="JSON metrics")
    metrics_parser.set_defaults(func=add_metrics)
    
    # Promote model
    promote_parser = subparsers.add_parser("promote", help="Promote model to production")
    promote_parser.add_argument("--model-id", required=True, help="Model ID")
    promote_parser.add_argument("--version", required=True, help="Model version")
    promote_parser.set_defaults(func=promote_model)
    
    # Sync model from Ollama
    sync_parser = subparsers.add_parser("sync", help="Sync a model from Ollama")
    sync_parser.add_argument("--model-name", required=True, help="Ollama model name (e.g. qwen3.5:4b)")
    sync_parser.add_argument("--version", help="Version override (auto-detected if omitted)")
    sync_parser.add_argument("--status", default="registered", choices=["registered", "staging", "production"])
    sync_parser.set_defaults(func=sync_model)
    
    # Sync all Ollama models
    sync_all_parser = subparsers.add_parser("sync-all", help="Sync all local Ollama models")
    sync_all_parser.set_defaults(func=sync_all)
    
    args = parser.parse_args()
    
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()