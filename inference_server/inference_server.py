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
import json
import os
import shutil
import tempfile
import uuid
import sys
from pathlib import Path
from dataclasses import asdict
from typing import Optional

# Add project root (AssetsAnything) to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("SPCONV_ALGO", "native")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from preprocessing.restructure_prompt import build_user_input, rewrite_prompt
from preprocessing.web_scraping import download_images, get_result
from preprocessing.dimension_lookup import lookup_dimensions_from_url
from preprocessing.image_filter import filter_candidates, segment_image
from preprocessing.image_gen import generate_reference_image
from postprocessing.scaling import scale_glb_to_real_world
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

# Where user-uploaded images are briefly hosted so Google Lens (via SerpApi)
# can fetch them by URL -- Lens only accepts public image URLs, not raw
# uploads. This directory is served at GET /tmp_uploads/{filename} below.
# Requires this server to actually be reachable from the internet when
# /identify_dimensions is called (an SSH-tunnel-only setup won't work for
# this one endpoint, even though everything else works fine over a tunnel).
TMP_UPLOAD_DIR = tempfile.mkdtemp(prefix="trellis_tmp_uploads_")

# VLM endpoint used for prompt rewriting/classification — point this at
# wherever you're serving Qwen2.5-VL (or another OpenAI-compatible VLM).
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://localhost:9000/v1")
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "")

# Optional -- set this if the server sits behind a reverse proxy / different
# public hostname than the one the request came in on. If unset, the
# request's own base URL is used (fine for a directly-exposed server).
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "")

# Backend used for the "AI-generated reference image" fallback in flow B,
# when scraping can't find enough consistent real photos. Point it at
# whatever image-gen backend exposes an OpenAI-compatible images API.
IMAGE_GEN_BASE_URL = os.environ.get("IMAGE_GEN_BASE_URL", "")
IMAGE_GEN_MODEL = os.environ.get("IMAGE_GEN_MODEL", "")

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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    multiimage_algo: str = Form("stochastic"),  # "stochastic" or "multidiffusion"
    target_dimensions_cm: str = Form(""),  # JSON string, e.g. {"length_cm":30,"width_cm":20,"height_cm":15}
    dimensions_source: str = Form("manual"),  # "manual" or "auto_detected", just for labeling the response
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

    result = run_trellis_and_export(pil_images, multiimage_algo)
    apply_optional_scaling(result, target_dimensions_cm, dimensions_source)
    return result


def apply_optional_scaling(result: dict, target_dimensions_cm_json: str, dimensions_source: str) -> None:
    """
    Mutates `result` in place: if a usable target dimension was supplied
    (auto-detected via /identify_dimensions or typed in manually on the
    client), rescales the exported .glb and swaps in the real-world
    bounding box. Leaves `result` untouched (normalized units) otherwise.
    """
    result["dimensions_source"] = "none"
    if not target_dimensions_cm_json:
        return

    try:
        target_dims = json.loads(target_dimensions_cm_json)
    except Exception:
        return

    out_path = os.path.join(OUTPUT_DIR, f"{result['job_id']}.glb")
    scaled = scale_glb_to_real_world(out_path, result["dimensions"], target_dims)
    if scaled:
        result["dimensions"] = scaled
        result["dimensions_source"] = dimensions_source


@app.get("/tmp_uploads/{filename}")
def serve_tmp_upload(filename: str):
    """
    Briefly serves a user-uploaded image so Google Lens (via SerpApi) can
    fetch it by URL -- see /identify_dimensions and TMP_UPLOAD_DIR above.
    """
    path = os.path.join(TMP_UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Not found (it may have already been cleaned up).")
    return FileResponse(path)


@app.post("/identify_dimensions")
async def identify_dimensions(
    request: Request,
    image: UploadFile = File(...),
    object_name_hint: str = Form(""),
):
    """
    Flow-A "auto-detect dimensions" checkbox: identifies the uploaded
    object via Google Lens, searches for its published specs via Google
    Search, and asks the LLM to extract a structured length/width/height.
    Returns found=false (never a guessed number) if nothing explicit was
    found -- the client should fall back to asking the user to type
    dimensions in by hand.
    """
    if not SERPAPI_API_KEY:
        raise HTTPException(500, "SERPAPI_API_KEY is not configured on the server.")

    raw = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    filename = f"{uuid.uuid4()}.jpg"
    os.makedirs(TMP_UPLOAD_DIR, exist_ok=True)
    tmp_path = os.path.join(TMP_UPLOAD_DIR, filename)
    pil_image.save(tmp_path, format="JPEG")

    base_url = PUBLIC_BASE_URL.rstrip("/") if PUBLIC_BASE_URL else str(request.base_url).rstrip("/")
    image_url = f"{base_url}/tmp_uploads/{filename}"

    result = lookup_dimensions_from_url(
        image_url, object_name_hint, SERPAPI_API_KEY, VLM_BASE_URL, VLM_MODEL
    )
    return asdict(result)


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
    object_name: str = ""  # from /rewrite_prompt's object_name -- used for CLIP filtering + dimension lookup
    prefer_shopping_results: bool = False
    multiimage_algo: str = "stochastic"
    auto_dimensions: bool = True
    target_dimensions_cm: Optional[dict] = None  # manual override; wins over auto-detection if provided


@app.post("/search_and_generate")
def search_and_generate(req: SearchAndGenerateRequest):
    """
    Takes a FINAL, already-resolved search query (i.e. after any
    clarification round-trip via /rewrite_prompt has completed) and runs
    the full flow-B pipeline:

      scrape candidates -> CLIP relevance + color-consistency filter
        -> if enough good images survive: segment each, optionally look up
           real-world dimensions via Google Lens + Search on the best one
        -> if not enough survive: generate a single AI reference image
           instead (dimension lookup is skipped -- there's no real object
           to identify)
      -> run TRELLIS -> optionally scale to real-world size -> export.
    """
    engine = "google_images"

    os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
    os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto'

    object_name = req.object_name or req.search_query

    scraped = get_result(req.search_query, engine=engine, language="en")
    if not scraped:
        raise HTTPException(
            422,
            f"No usable images found for {req.search_query!r}. "
            "Try a more specific prompt.",
        )

    job_dir = os.path.join(SCRAPE_DIR, str(uuid.uuid4()))
    downloaded = download_images(scraped, job_dir)
    if not downloaded:
        raise HTTPException(422, "Found candidate images but all downloads failed.")

    candidates = []  # list of (pil_image, ScrapedImage)
    for path, scraped_img in downloaded:
        try:
            candidates.append((Image.open(path).convert("RGBA"), scraped_img))
        except Exception as e:
            print(f"[server] skipping unreadable image {path}: {e}")

    if not candidates:
        raise HTTPException(422, "Downloaded files could not be read as images.")

    kept_indices, filtering_ok = filter_candidates([im for im, _ in candidates], object_name)
    used_ai_reference = False
    identified_dimensions = None

    if filtering_ok:
        kept = [candidates[i] for i in kept_indices]
        pil_images = [segment_image(im) for im, _ in kept]

        if req.auto_dimensions and SERPAPI_API_KEY:
            best_image_url = kept[0][1].url
            dim_result = lookup_dimensions_from_url(
                best_image_url, object_name, SERPAPI_API_KEY, VLM_BASE_URL, VLM_MODEL
            )
            if dim_result.found:
                identified_dimensions = asdict(dim_result)
    else:
        used_ai_reference = True
        if not (IMAGE_GEN_BASE_URL and IMAGE_GEN_MODEL):
            raise HTTPException(
                422,
                f"No consistent, on-topic photos of {object_name!r} were found, and no "
                "IMAGE_GEN_BASE_URL/IMAGE_GEN_MODEL is configured for the AI-generated "
                "fallback. Try a more specific prompt, or configure an image-gen backend.",
            )
        pil_images = [generate_reference_image(object_name, IMAGE_GEN_BASE_URL, IMAGE_GEN_MODEL)]

    result = run_trellis_and_export(pil_images, req.multiimage_algo)
    result["num_images_used"] = len(pil_images)
    result["used_ai_generated_reference"] = used_ai_reference
    result["source_titles"] = (
        ["AI-generated reference image"] if used_ai_reference
        else [candidates[i][1].title for i in kept_indices]
    )

    target_dims = req.target_dimensions_cm
    dimensions_source = "manual"
    if not target_dims and identified_dimensions and identified_dimensions.get("found"):
        target_dims = {
            "length_cm": identified_dimensions.get("length_cm"),
            "width_cm": identified_dimensions.get("width_cm"),
            "height_cm": identified_dimensions.get("height_cm"),
        }
        dimensions_source = "auto_detected"

    result["identified_dimensions"] = identified_dimensions
    result["dimensions_source"] = "none"
    if target_dims:
        scaled = scale_glb_to_real_world(
            os.path.join(OUTPUT_DIR, f"{result['job_id']}.glb"), result["dimensions"], target_dims
        )
        if scaled:
            result["dimensions"] = scaled
            result["dimensions_source"] = dimensions_source

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
