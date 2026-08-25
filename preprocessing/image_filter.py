"""
Filters scraped image candidates down to a consistent, on-topic set before
they're fed to TRELLIS: a CLIP relevance pass (drop images that don't
actually match the object), then a color-consistency pass (drop outliers,
since a query like "Nike Air Jordan 1" will scrape several colorways and
multi-image fusion assumes ALL views are of the SAME physical object).

Both passes degrade gracefully: if a dependency (open_clip / rembg) isn't
installed on the remote box, we log a warning and skip that pass rather
than failing the whole request.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

log = logging.getLogger("image_filter")

CLIP_RELEVANCE_THRESHOLD = 0.20   # cosine similarity, OpenCLIP-typical scale
COLOR_OUTLIER_DISTANCE = 60.0     # euclidean distance in mean-RGB space (0-255)
MIN_KEPT_IMAGES = 1               # below this, caller should treat the batch as "bad"

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def _get_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        import open_clip  # pip install open_clip_torch
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        _clip_model, _clip_preprocess, _clip_tokenizer = model, preprocess, tokenizer
    return _clip_model, _clip_preprocess, _clip_tokenizer


def clip_relevance_scores(images: list[Image.Image], query_text: str) -> Optional[list[float]]:
    """Cosine similarity of each image to `query_text`. None if CLIP is unavailable."""
    try:
        model, preprocess, tokenizer = _get_clip()
        import torch

        device = next(model.parameters()).device
        with torch.no_grad():
            image_tensors = torch.stack([preprocess(im.convert("RGB")) for im in images]).to(device)
            image_features = model.encode_image(image_tensors)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            text_tokens = tokenizer([query_text]).to(device)
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            sims = (image_features @ text_features.T).squeeze(-1)
        return [float(s) for s in sims.cpu()]
    except Exception as e:
        log.warning(f"CLIP relevance filtering unavailable, skipping ({e})")
        return None


def _mean_rgb(image: Image.Image) -> np.ndarray:
    small = image.convert("RGB").resize((32, 32))
    return np.asarray(small, dtype=np.float32).reshape(-1, 3).mean(axis=0)


def color_consistency_mask(images: list[Image.Image]) -> list[bool]:
    """
    Flags images whose mean color is far from the group's median color as
    outliers (likely a different colorway/variant scraped under the same
    query). Always "succeeds" — pure numpy, no optional dependency.
    """
    if len(images) <= 2:
        return [True] * len(images)

    colors = np.stack([_mean_rgb(im) for im in images])
    median_color = np.median(colors, axis=0)
    distances = np.linalg.norm(colors - median_color, axis=1)
    return [bool(d <= COLOR_OUTLIER_DISTANCE) for d in distances]


def filter_candidates(images: list[Image.Image], object_name: str) -> tuple[list[int], bool]:
    """
    Runs CLIP relevance + color-consistency filtering over `images`.
    Returns (kept_indices, ok) where `ok` is False if too few images
    survived filtering and the caller should fall back to an AI-generated
    reference image instead.
    """
    n = len(images)
    keep = list(range(n))

    clip_scores = clip_relevance_scores(images, object_name)
    if clip_scores is not None:
        keep = [i for i in keep if clip_scores[i] >= CLIP_RELEVANCE_THRESHOLD]
        if not keep:
            # CLIP rejected everything -- relax rather than nuke the whole batch,
            # keep the single best-scoring image instead.
            keep = [int(np.argmax(clip_scores))]

    consistent = color_consistency_mask([images[i] for i in keep])
    kept = [idx for idx, ok in zip(keep, consistent) if ok]

    if len(kept) < MIN_KEPT_IMAGES:
        return kept, False
    return kept, True


def segment_image(image: Image.Image) -> Image.Image:
    """Removes the background so TRELLIS gets an isolated subject on transparent RGBA."""
    try:
        from rembg import remove  # pip install rembg

        return remove(image.convert("RGBA"))
    except Exception as e:
        log.warning(f"Background removal unavailable, using image as-is ({e})")
        return image.convert("RGBA")
