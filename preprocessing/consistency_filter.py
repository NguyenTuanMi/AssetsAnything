"""
Consistency filter: given a batch of candidate images for the same search
query, find the largest subset that plausibly depicts the SAME physical
object/variant (not just the same product category) — before handing them
to TRELLIS for multi-view reconstruction.

Two complementary signals, since they catch different failure modes:

  1. CLIP embedding similarity — catches "wrong object entirely" outliers
     (a shoe box, a lifestyle photo, an unrelated product). CLIP is
     semantic/category-level, so it is comparatively weak at telling apart
     visually-similar variants (e.g. two colorways of the same sneaker).

  2. Color histogram similarity (on the foreground only, post background
     removal) — catches "right object, wrong colorway" outliers, which
     CLIP alone tends to miss.

Usage:
    from consistency_filter import select_consistent_images
    kept_paths = select_consistent_images(image_paths)
"""

import logging

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("consistency_filter")

_clip_model = None
_clip_preprocess = None
_clip_device = None


def _load_clip():
    """Loads CLIP once, lazily, so importing this module doesn't require a GPU."""
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is not None:
        return

    import torch
    from transformers import CLIPModel, CLIPProcessor

    _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_clip_device)
    _clip_model.eval()
    _clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    log.info(f"CLIP loaded on {_clip_device}")


def compute_clip_embeddings(image_paths: list[str]) -> np.ndarray:
    """Returns an (N, D) L2-normalized embedding matrix, one row per image."""
    import torch

    _load_clip()
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = _clip_preprocess(images=images, return_tensors="pt").to(_clip_device)

    with torch.no_grad():
        features = _clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy()


def compute_color_histograms(image_paths: list[str], bins: int = 32) -> np.ndarray:
    """
    Returns an (N, bins*3) matrix of normalized HSV histograms — one per
    image. Uses the full image (not background-removed) for simplicity;
    if you already run background removal upstream, pass those paths in
    instead for a cleaner foreground-only signal.
    """
    hists = []
    for p in image_paths:
        img = Image.open(p).convert("HSV")
        arr = np.array(img)
        h_hist, _ = np.histogram(arr[:, :, 0], bins=bins, range=(0, 255), density=True)
        s_hist, _ = np.histogram(arr[:, :, 1], bins=bins, range=(0, 255), density=True)
        v_hist, _ = np.histogram(arr[:, :, 2], bins=bins, range=(0, 255), density=True)
        hists.append(np.concatenate([h_hist, s_hist, v_hist]))
    return np.array(hists)


def _cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return norm @ norm.T


def _histogram_intersection_matrix(hists: np.ndarray) -> np.ndarray:
    """Histogram intersection similarity (1.0 = identical, 0.0 = disjoint)."""
    n = hists.shape[0]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = np.minimum(hists[i], hists[j]).sum()
    return sim


def _largest_consistent_cluster(
    combined_sim: np.ndarray,
    threshold: float,
) -> list[int]:
    """
    Builds a graph where an edge connects two images if their combined
    similarity exceeds `threshold`, then returns the indices of the
    largest connected component. This directly implements "keep the
    largest group of mutually-agreeing images, drop the outliers."
    """
    n = combined_sim.shape[0]
    adjacency = combined_sim >= threshold
    np.fill_diagonal(adjacency, False)

    visited = set()
    best_component = []

    for start in range(n):
        if start in visited:
            continue
        stack = [start]
        component = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            neighbors = np.where(adjacency[node])[0]
            stack.extend(n for n in neighbors if n not in visited)
        if len(component) > len(best_component):
            best_component = component

    return sorted(best_component)


def select_consistent_images(
    image_paths: list[str],
    clip_weight: float = 0.6,
    color_weight: float = 0.4,
    similarity_threshold: float = 0.75,
    min_cluster_size: int = 2,
) -> list[str]:
    """
    Main entry point. Returns the subset of image_paths that form the
    largest mutually-consistent group — same object AND same variant/
    colorway — dropping outliers.

    If the largest consistent cluster is smaller than min_cluster_size,
    returns an empty list, signaling "not enough consistent images found"
    so the caller can fail fast rather than force a bad TRELLIS run.
    """
    if len(image_paths) < 2:
        log.warning("Fewer than 2 images provided — nothing to cluster.")
        return image_paths

    clip_embeds = compute_clip_embeddings(image_paths)
    color_hists = compute_color_histograms(image_paths)

    clip_sim = _cosine_sim_matrix(clip_embeds)
    color_sim = _histogram_intersection_matrix(color_hists)

    combined = clip_weight * clip_sim + color_weight * color_sim

    cluster_indices = _largest_consistent_cluster(combined, similarity_threshold)

    log.info(
        f"Largest consistent cluster: {len(cluster_indices)}/{len(image_paths)} images "
        f"(threshold={similarity_threshold})"
    )

    if len(cluster_indices) < min_cluster_size:
        log.warning(
            f"Largest consistent cluster ({len(cluster_indices)}) is below "
            f"min_cluster_size ({min_cluster_size}) — treat as a failed scrape."
        )
        return []

    return [image_paths[i] for i in cluster_indices]


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Filter images to the largest consistent group")
    parser.add_argument("image_paths", nargs="+", help="Paths to candidate images")
    parser.add_argument("--clip_weight", type=float, default=0.6)
    parser.add_argument("--color_weight", type=float, default=0.4)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--min_cluster_size", type=int, default=2)
    args = parser.parse_args()

    kept = select_consistent_images(
        args.image_paths,
        clip_weight=args.clip_weight,
        color_weight=args.color_weight,
        similarity_threshold=args.threshold,
        min_cluster_size=args.min_cluster_size,
    )
    print(json.dumps(kept, indent=2))


if __name__ == "__main__":
    main()