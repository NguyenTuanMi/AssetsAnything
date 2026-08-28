"""
Post-processing step: rescales a TRELLIS-exported .glb from its arbitrary
normalized model-space size to real-world centimeters.

TRELLIS has no notion of true physical scale and no reliable way to know
which bounding-box axis (X/Y/Z) corresponds to which real-world dimension
(length/width/height) a product page reports -- reported axis order varies
by source and TRELLIS's own output orientation isn't guaranteed to match
it. The default (`uniform=True`, used by auto-detected dimensions) sidesteps
that by applying a single scale factor: the largest reported real-world
dimension is matched to the largest bounding-box axis extent, fixing
absolute scale while preserving whatever proportions TRELLIS already got
right.

Manual resize (`uniform=False`) trades that safety for control: the user
typed three specific numbers and wants the model stretched to hit them
exactly, proportions be damned. It maps each target value to a bounding-box
axis by rank (largest target -> whichever axis is currently largest, and so
on) and scales each axis independently.
"""
from __future__ import annotations

from typing import Optional


def scale_glb_to_real_world(
    glb_path: str,
    current_bbox: dict,
    target_dims_cm: dict,
    uniform: bool = True,
) -> Optional[dict]:
    """
    Rescales the mesh at `glb_path` in place.

    `current_bbox` is the dict returned by compute_bounding_box() (normalized
    model-space units). `target_dims_cm` may contain any subset of
    {"length_cm", "width_cm", "height_cm"}.

    If `uniform` is True (default), only the largest present target value is
    used and the same factor is applied to all three axes -- safe when the
    values came from auto-detection and axis correspondence is unknown.

    If `uniform` is False, all three target values are used and each is
    matched to a bounding-box axis by rank, scaling non-uniformly -- for
    deliberate manual resizing where the user wants those exact dimensions
    regardless of the model's original proportions. Requires all three
    values present and positive.

    Returns the recomputed bounding box (in centimeters) or None if the
    required target values weren't usable.
    """
    import trimesh

    axis_names = ["width_x", "height_y", "depth_z"]
    current_extents = [current_bbox[a] for a in axis_names]

    if uniform:
        target_values = [v for v in target_dims_cm.values() if isinstance(v, (int, float)) and v > 0]
        if not target_values:
            return None
        target_max_cm = max(target_values)

        current_max = max(current_extents)
        if current_max <= 0:
            return None

        scale_vector = [target_max_cm / current_max] * 3
    else:
        target_values = [v for v in target_dims_cm.values() if isinstance(v, (int, float)) and v > 0]
        if len(target_values) < 3 or any(e <= 0 for e in current_extents):
            return None

        target_sorted = sorted(target_values, reverse=True)
        axis_order_by_size = sorted(range(3), key=lambda i: current_extents[i], reverse=True)

        scale_vector = [1.0, 1.0, 1.0]
        for rank, axis_idx in enumerate(axis_order_by_size):
            scale_vector[axis_idx] = target_sorted[rank] / current_extents[axis_idx]

    scene_or_mesh = trimesh.load(glb_path, force="scene")
    scene_or_mesh.apply_scale(scale_vector)
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
        "scale_factor_applied": round(float(scale_vector[0]), 6) if uniform else [round(float(s), 6) for s in scale_vector],
    }
