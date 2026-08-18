"""
TRELLIS inference server.

Runs on your remote GPU machine (inside the `trellis` conda env).
Exposes a single endpoint that accepts multiple images and returns a .glb file.

Start it with:
    conda activate trellis
    pip install fastapi "uvicorn[standard]" python-multipart
    python inference_server.py --host 0.0.0.0 --port 8000

Then, from your laptop, either:
  - open the port on your firewall / cloud security group, or
  - SSH-tunnel it:  ssh -L 8000:localhost:8000 user@remote-server
"""

import argparse
import io
import os
import shutil
import tempfile
import uuid
import sys
from pathlib import Path
from dataclasses import asdict

# Add project root (AssetsAnything) to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("SPCONV_ALGO", "native")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from preprocessing.restructure_prompt import build_user_input, rewrite_prompt
from preprocessing.web_scraping import download_images, get_result
from PIL import Image

app = FastAPI(title="TRELLIS Inference Server")

# Allow the local client (served from file:// or localhost) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup, reused across requests (loading the pipeline is slow).
pipeline = None

# Where finished .glb files are written, keyed by job id, so the client can fetch them.
OUTPUT_DIR = tempfile.mkdtemp(prefix="trellis_outputs_")

SCRAPE_DIR = tempfile.mkdtemp(prefix="trellis_scrapes_")
 
# VLM endpoint used for prompt rewriting/classification — point this at
# wherever you're serving Qwen2.5-VL (or another OpenAI-compatible VLM).
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://localhost:9000/v1")
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

def get_pipeline():
    global pipeline
    if pipeline is None:
        from submodules.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline

        print("[server] loading TRELLIS pipeline (first request only)...")
        pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")

        pipeline.cuda()
        print("[server] pipeline ready.")
    return pipeline

def run_trellis_and_export(pil_images: list[Image.Image], multiimage_algo: str = "stochastic") -> dict:
    """
    Shared core: runs TRELLIS on a list of PIL images (however they were
    sourced — direct upload or scraped), exports a .glb, and computes its
    bounding box. Used by both /generate (uploads) and /search_and_generate
    (prompt + scraping) so the actual inference logic lives in one place.
    """
    pipe = get_pipeline()
 
    from submodules.TRELLIS.trellis.utils import postprocessing_utils
    print(f"[server] running pipeline on {len(pil_images)} images "
          f"(algo={multiimage_algo})...")
 
    if len(pil_images) == 1:
        outputs = pipe.run(pil_images[0], seed=1)
    else:
        outputs = pipe.run_multi_image(
            pil_images,
            seed=1,
            mode=multiimage_algo,
        )
 
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=0.95,
        texture_size=1024,
    )
 
    job_id = str(uuid.uuid4())
    out_path = os.path.join(OUTPUT_DIR, f"{job_id}.glb")
    glb.export(out_path)
 
    dimensions = compute_bounding_box(out_path)
 
    print(f"[server] done -> {out_path}  bbox={dimensions}")
    return {
        "job_id": job_id,
        "download_url": f"/download/{job_id}",
        "dimensions": dimensions,
    }


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}


@app.post("/generate")
async def generate(
    images: list[UploadFile] = File(...),
    multiimage_algo: str = "stochastic",  # "stochastic" or "multidiffusion"
):
    os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
    os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
    if not images:
        raise HTTPException(400, "No images uploaded.")

    pil_images = []
    for f in images:
        raw = await f.read()
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGBA")
        except Exception as e:
            raise HTTPException(400, f"Could not read image '{f.filename}': {e}")
        pil_images.append(img)

    return run_trellis_and_export(pil_images, multiimage_algo)

class QAPair(BaseModel):
    question: str
    answer: str
 
 
class RewritePromptRequest(BaseModel):
    prompt: str
    qa_history: list[QAPair] = []

@app.post("/rewrite_prompt")
def rewrite_prompt_endpoint(req: RewritePromptRequest):
    """
    Turns a user's raw prompt into a structured search query, or asks a
    clarifying question. This endpoint does NOT loop internally — the
    "reprompt" behavior lives on the client: if classification comes back
    as "needs_clarification", the client shows clarification_question
    (and candidate_interpretations, if any) in a question box, collects
    the user's answer, and calls this endpoint again with that Q&A pair
    appended to qa_history. Repeat until classification resolves to
    "specific_product" or "generic_concept".
    """
    augmented_input = build_user_input(
        req.prompt, [qa.model_dump() for qa in req.qa_history]
    )
    result = rewrite_prompt(
        user_prompt=augmented_input,
        vlm_base_url=VLM_BASE_URL,
        vlm_model=VLM_MODEL,
    )
 
    if result.classification == "error":
        raise HTTPException(502, f"Prompt rewriting failed: {result.error}")
 
    return asdict(result)
 
 
class SearchAndGenerateRequest(BaseModel):
    search_query: str
    prefer_shopping_results: bool = False
    multiimage_algo: str = "stochastic"

@app.post("/search_and_generate")
def search_and_generate(req: SearchAndGenerateRequest):
    """
    Takes a FINAL, already-resolved search query (i.e. after any
    clarification round-trip via /rewrite_prompt has completed) — searches
    for candidate images, downloads them, and runs them straight through
    TRELLIS. Mirrors /generate's response shape.
    """
    engine = "google_images"

    os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
    os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto'
 
    scraped = get_result(req.search_query, engine=engine, language="en")
    if not scraped:
        raise HTTPException(
            422,
            f"No usable images found for {req.search_query!r}. "
            "Try a more specific prompt.",
        )
 
    job_dir = os.path.join(SCRAPE_DIR, str(uuid.uuid4()))
    local_paths = download_images(scraped, job_dir)
    if not local_paths:
        raise HTTPException(422, "Found candidate images but all downloads failed.")
 
    pil_images = []
    for p in local_paths:
        try:
            pil_images.append(Image.open(p).convert("RGBA"))
        except Exception as e:
            print(f"[server] skipping unreadable image {p}: {e}")
 
    if not pil_images:
        raise HTTPException(422, "Downloaded files could not be read as images.")
 
    result = run_trellis_and_export(pil_images, req.multiimage_algo)
    result["num_images_used"] = len(pil_images)
    result["source_titles"] = [img.title for img in scraped[:len(pil_images)]]
    return result

def compute_bounding_box(glb_path: str):
    """
    Reload the exported .glb with trimesh and compute its axis-aligned
    bounding box. Returns size (width/height/depth), min/max corners, and
    diagonal length, all in TRELLIS's normalized model-space units (not
    real-world units — TRELLIS doesn't know true physical scale).
    """
    import trimesh

    scene_or_mesh = trimesh.load(glb_path, force="scene")

    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = scene_or_mesh.to_geometry() if len(scene_or_mesh.geometry) else None
        bounds = scene_or_mesh.bounds if mesh is None else mesh.bounds
    else:
        bounds = scene_or_mesh.bounds

    min_corner, max_corner = bounds[0], bounds[1]
    size = max_corner - min_corner

    return {
        "width_x": round(float(size[0]), 4),
        "height_y": round(float(size[1]), 4),
        "depth_z": round(float(size[2]), 4),
        "diagonal": round(float((size ** 2).sum() ** 0.5), 4),
        "min_corner": [round(float(v), 4) for v in min_corner],
        "max_corner": [round(float(v), 4) for v in max_corner],
        "units": "normalized (model space, not real-world scale)",
    }

@app.get("/download/{job_id}")
def download(job_id: str):
    path = os.path.join(OUTPUT_DIR, f"{job_id}.glb")
    if not os.path.exists(path):
        raise HTTPException(404, "File not found (server may have restarted).")
    return FileResponse(path, media_type="model/gltf-binary", filename=f"{job_id}.glb")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
