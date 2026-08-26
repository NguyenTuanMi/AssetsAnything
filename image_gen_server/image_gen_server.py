"""
Image-generation server for the flow-B "AI-generated reference image"
fallback (used when scraping can't find enough consistent, on-topic
photos). Exposes a minimal OpenAI-compatible /v1/images/generations
endpoint so preprocessing/image_gen.py's OpenAI client can talk to it
unmodified.

Runs on its own dedicated GPU, separate from the TRELLIS process and the
Qwen VLM process.

Start it pinned to whichever GPU is free (you're running one process per
GPU across TRELLIS / Qwen-VLM / this):
    conda activate <your env with diffusers/torch>
    pip install fastapi "uvicorn[standard]" diffusers transformers accelerate safetensors torch
    CUDA_VISIBLE_DEVICES=2 python image_gen_server.py --host 0.0.0.0 --port 8100

Then point the inference server at it:
    export IMAGE_GEN_BASE_URL=http://<this-machine>:8100/v1
    export IMAGE_GEN_MODEL=stabilityai/sdxl-turbo
"""
import argparse
import base64
import io
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Image Generation Server")

MODEL_ID = os.environ.get("IMAGE_GEN_MODEL_ID", "stabilityai/sdxl-turbo")

pipe = None


def get_pipe():
    """Loaded once at startup (first request), reused across requests."""
    global pipe
    if pipe is None:
        import torch
        from diffusers import AutoPipelineForText2Image

        print(f"[image-gen] loading {MODEL_ID}...")
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, variant="fp16"
        )
        pipe.to("cuda")
        print("[image-gen] ready.")
    return pipe


class ImageGenRequest(BaseModel):
    model: str | None = None
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipe is not None, "model_id": MODEL_ID}


@app.post("/v1/images/generations")
def generate(req: ImageGenRequest):
    if req.response_format != "b64_json":
        raise HTTPException(400, "Only response_format='b64_json' is supported.")

    p = get_pipe()

    try:
        target_w, target_h = (int(x) for x in req.size.lower().split("x"))
    except Exception:
        target_w, target_h = 1024, 1024

    data = []
    for _ in range(max(1, req.n)):
        # SDXL-Turbo: single-step, no negative prompt, guidance_scale must be 0.
        image = p(
            prompt=req.prompt,
            num_inference_steps=1,
            guidance_scale=0.0,
        ).images[0]

        if image.size != (target_w, target_h):
            image = image.resize((target_w, target_h))

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data.append({"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")})

    return {"created": 0, "data": data}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
