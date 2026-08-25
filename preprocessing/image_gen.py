"""
Fallback path for flow B: when scraping couldn't find enough consistent,
on-topic photos of the described object (CLIP+color filtering rejected
the batch), generate a single clean reference image instead of failing
the request outright.

Point IMAGE_GEN_BASE_URL at whatever image-generation backend you're
self-hosting (anything exposing an OpenAI-compatible `/v1/images/generations`
route -- e.g. a ComfyUI/SD WebUI shim, or the real OpenAI API). This is a
thin, swappable client, not a specific model choice.
"""
from __future__ import annotations

import base64
import io
import logging

from PIL import Image

log = logging.getLogger("image_gen")

REFERENCE_IMAGE_PROMPT = (
    "{object_name}, isolated product photo, plain white background, "
    "studio lighting, single object, no text, no watermark, centered"
)


def generate_reference_image(
    object_name: str,
    base_url: str,
    model: str,
    api_key: str = "not-needed",
) -> Image.Image:
    """Generates one clean product-style photo to use as the sole TRELLIS input."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    prompt = REFERENCE_IMAGE_PROMPT.format(object_name=object_name)

    response = client.images.generate(
        model=model,
        prompt=prompt,
        n=1,
        size="1024x1024",
        response_format="b64_json",
    )
    b64 = response.data[0].b64_json
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
