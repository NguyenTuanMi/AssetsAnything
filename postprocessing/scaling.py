"""
Post-processing step: rescales a TRELLIS-exported .glb from its arbitrary
normalized model-space size to real-world centimeters.

TRELLIS has no notion of true physical scale and no reliable way to know
which bounding-box axis (X/Y/Z) corresponds to which real-world dimension
(length/width/height) a product page reports -- reported axis order varies
by source and TRELLIS's own output orientation isn't guaranteed to match
it. Rather than guess a per-axis mapping and risk warping proportions,
this applies a single uniform scale factor: the largest reported
real-world dimension is matched to the largest bounding-box axis extent.
This fixes absolute scale while preserving whatever proportions TRELLIS
already got right.
"""
from __future__ import annotations

from typing import Optional


def scale_glb_to_real_world(
    glb_path: str,
    current_bbox: dict,
    target_dims_cm: dict,
) -> Optional[dict]:
    """
    Rescales the mesh at `glb_path` in place.

    `current_bbox` is the dict returned by compute_bounding_box() (normalized
    model-space units). `target_dims_cm` may contain any subset of
    {"length_cm", "width_cm", "height_cm"} -- only the largest present value
    is used, since only a uniform scale is applied.

    Returns the recomputed bounding box (in centimeters) or None if
    `target_dims_cm` had no usable numbers.
    """
    import trimesh

    target_values = [v for v in target_dims_cm.values() if isinstance(v, (int, float)) and v > 0]
    if not target_values:
        return None
    target_max_cm = max(target_values)

    current_max = max(current_bbox["width_x"], current_bbox["height_y"], current_bbox["depth_z"])
    if current_max <= 0:
        return None

    scale_factor = target_max_cm / current_max

    scene_or_mesh = trimesh.load(glb_path, force="scene")
    scene_or_mesh.apply_scale(scale_factor)
    scene_or_mesh.export(glb_path)

    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = scene_or_mesh.to_geometry() if len(scene_or_mesh.geometry) else None
        bounds = scene_or_mesh.bounds if mesh is None else mesh.bounds
    else:
        bounds = scene_or_mesh.bounds

    min_corner, max_corner = bounds[0], bounds[1]
    size = max_corner - min_corner

    return {
        "width_x": round(float(size[0]), 3),
        "height_y": round(float(size[1]), 3),
        "depth_z": round(float(size[2]), 3),
        "diagonal": round(float((size ** 2).sum() ** 0.5), 3),
        "min_corner": [round(float(v), 3) for v in min_corner],
        "max_corner": [round(float(v), 3) for v in max_corner],
        "units": "centimeters (scaled from an auto-detected or user-entered real-world dimension)",
        "scale_factor_applied": round(float(scale_factor), 6),
    }
