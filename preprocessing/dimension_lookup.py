"""
Real-world dimension lookup: identify an object from a photo (Google Lens,
via SerpApi), search for its published spec sheet (Google Search, via
SerpApi), then force the search results through an LLM to pull out a
structured length/width/height.

Google Lens (as exposed by SerpApi) only accepts a *public* image URL, not
a raw upload — the caller is responsible for making sure `image_url` is
actually reachable from the internet (a scraped image's original URL
already is; a user-uploaded image needs to be temporarily hosted first,
see inference_server.py's /identify_dimensions).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import serpapi

log = logging.getLogger("dimension_lookup")

EXTRACTION_SYSTEM_PROMPT = """
You are extracting physical product dimensions from noisy web search results.

You will be given a product name and a set of search result snippets/titles.
Find the object's real-world length, width, and height (or diameter, for
round objects — put it in "width" and leave "length" null).

STRICT RULES:
1. Only use a number if it is EXPLICITLY stated in the provided text. Never
   estimate, infer from category averages, or guess a "plausible" size.
2. If the snippets don't contain an explicit dimension, set "found" to
   false and leave the dimension fields null. Do not make one up.
3. Convert everything to centimeters. If the source used inches, multiply
   by 2.54. If it used millimeters, divide by 10.
4. If multiple conflicting values appear, prefer the one attached to the
   most authoritative-looking source (an official manufacturer/retailer
   product page over a forum/blog), and reflect the disagreement by
   lowering "confidence".

Output ONLY valid JSON in this exact schema, no other text:
{{
  "found": boolean,
  "length_cm": number | null,
  "width_cm": number | null,
  "height_cm": number | null,
  "confidence": "high" | "medium" | "low",
  "source_url": string | null
}}

Product name: {object_name}

Search results:
{search_context}
"""


@dataclass
class DimensionResult:
    identified_name: str
    found: bool
    length_cm: Optional[float] = None
    width_cm: Optional[float] = None
    height_cm: Optional[float] = None
    confidence: Optional[str] = None
    source_url: Optional[str] = None
    error: Optional[str] = None


def google_lens_identify(image_url: str, serpapi_key: str) -> Optional[str]:
    """
    Runs Google Lens on a publicly-reachable image URL and returns its best
    guess at what the object is (brand/product name), or None if Lens
    didn't return anything usable.
    """
    client = serpapi.Client(api_key=serpapi_key)
    try:
        results = client.search({"engine": "google_lens", "url": image_url})
    except Exception as e:
        log.warning(f"Google Lens call failed for {image_url}: {e}")
        return None

    # # SerpApi's google_lens response varies by result type; check the
    # # likely places for a title in priority order.
    # for key in ("knowledge_graph",):
    #     kg = results.get(key)
    #     if kg and isinstance(kg, list) and kg[0].get("title"):
    #         return kg[0]["title"]

    visual_matches = results.get("visual_matches", [])
    if visual_matches and visual_matches[0].get("title"):
        return visual_matches[0]["title"]

    return None


def google_search_specs(query: str, serpapi_key: str, max_results: int = 5) -> str:
    """
    Runs a plain Google search and flattens the top organic results (plus
    the knowledge graph / answer box, if present) into a text blob the LLM
    can read. Returns "" if nothing came back.
    """
    client = serpapi.Client(api_key=serpapi_key)
    try:
        results = client.search({"engine": "google", "q": query})
    except Exception as e:
        log.warning(f"Google search failed for {query!r}: {e}")
        return ""

    chunks = []

    kg = results.get("knowledge_graph")
    if kg:
        chunks.append(f"[Knowledge graph] {json.dumps(kg)[:1000]}")

    answer_box = results.get("answer_box")
    if answer_box:
        chunks.append(f"[Answer box] {json.dumps(answer_box)[:1000]}")

    for item in results.get("organic_results", [])[:max_results]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        chunks.append(f"[{link}] {title} — {snippet}")

    return "\n".join(chunks)


def extract_dimensions_llm(
    object_name: str,
    search_context: str,
    vlm_base_url: str,
    vlm_model: str,
    vlm_api_key: str = "not-needed",
) -> dict:
    """Text-only call to the same OpenAI-compatible endpoint used for prompt rewriting."""
    from urllib.parse import urlparse

    import httpx
    from openai import OpenAI

    host = urlparse(vlm_base_url).hostname
    is_local = host in ("localhost", "127.0.0.1", "::1")
    http_client = httpx.Client(trust_env=not is_local)
    client = OpenAI(base_url=vlm_base_url, api_key=vlm_api_key, http_client=http_client)

    prompt = EXTRACTION_SYSTEM_PROMPT.format(
        object_name=object_name, search_context=search_context or "(no search results found)"
    )
    response = client.chat.completions.create(
        model=vlm_model,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        temperature=0.0,
        max_tokens=300,
    )
    raw = response.choices[0].message.content.strip().strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    return json.loads(raw)


def lookup_dimensions_from_url(
    image_url: str,
    object_name_hint: str,
    serpapi_key: str,
    vlm_base_url: str,
    vlm_model: str,
) -> DimensionResult:
    """
    Full pipeline: Lens identify -> Search for specs -> LLM extraction.
    `object_name_hint` (e.g. from prompt rewriting, or a free-text hint the
    user typed) is used as a fallback/merge when Lens can't identify the
    object, and is always appended to the search query for specificity.
    """
    try:
        lens_name = google_lens_identify(image_url, serpapi_key)
        identified_name = lens_name or object_name_hint or "the object"

        query_bits = " ".join(x for x in [lens_name, object_name_hint] if x)
        search_context = google_search_specs(
            f"{query_bits} dimensions size specifications".strip(), serpapi_key
        )

        parsed = extract_dimensions_llm(identified_name, search_context, vlm_base_url, vlm_model)

        return DimensionResult(
            identified_name=identified_name,
            found=bool(parsed.get("found")),
            length_cm=parsed.get("length_cm"),
            width_cm=parsed.get("width_cm"),
            height_cm=parsed.get("height_cm"),
            confidence=parsed.get("confidence"),
            source_url=parsed.get("source_url"),
        )
    except Exception as e:
        log.warning(f"Dimension lookup failed: {e}")
        return DimensionResult(identified_name=object_name_hint or "the object", found=False, error=str(e))
