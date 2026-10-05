"""Model worker process for one AssetsAnything backend.

The management server starts this module with the model-specific interpreter.
Workers receive shared-filesystem paths over HTTP so model objects and CUDA
state never cross the environment boundary.
"""
from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from PIL import Image

from . import get_reconstructor, list_backends
from .base import ReconstructionRequest


class GenerateRequest(BaseModel):
    input_paths: list[str]
    output_path: str
    options: dict = Field(default_factory=dict)


def create_app(backend_name: str) -> FastAPI:
    app = FastAPI(title=f"AssetsAnything {backend_name} worker")
    inference_lock = threading.Lock()

    @app.get("/health")
    def health():
        loaded = next(
            (item for item in list_backends() if item["name"] == backend_name),
            None,
        )
        return {
            "status": "ok",
            "backend": backend_name,
            "model_loaded": bool(loaded and loaded["loaded"]),
        }

    @app.get("/capabilities")
    def capabilities():
        for item in list_backends():
            if item["name"] == backend_name:
                return item
        raise HTTPException(404, f"Unknown backend {backend_name!r}")

    @app.post("/generate")
    async def generate(request: GenerateRequest):
        if not request.input_paths:
            raise HTTPException(400, "At least one input image is required.")

        images = []
        for raw_path in request.input_paths:
            path = Path(raw_path).resolve()
            try:
                images.append(Image.open(path).convert("RGBA"))
            except Exception as error:
                raise HTTPException(400, f"Could not read image {path}: {error}") from error

        def run():
            with inference_lock:
                backend = get_reconstructor(backend_name)
                return backend.reconstruct(
                    ReconstructionRequest(
                        images=images,
                        output_path=str(Path(request.output_path).resolve()),
                        options=request.options,
                    )
                )

        try:
            result = await run_in_threadpool(run)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except Exception as error:
            raise HTTPException(500, str(error)) from error

        return {
            "backend": result.backend_name,
            "glb_path": result.glb_path,
            "num_images_used": result.num_images_used,
            "raw_meta": result.raw_meta,
        }

    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    uvicorn.run(create_app(args.backend), host=args.host, port=args.port)
