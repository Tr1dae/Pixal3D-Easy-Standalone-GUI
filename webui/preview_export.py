"""Lightweight WIP GLB exporters for live generation previews."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

# Same orientation used for final Pixal3D GLB export
_PREVIEW_ROT = np.array(
    [
        [-1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)

# +90° about X — upright in model-viewer for occupancy / clay WIP
_WIP_UP_ROT = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)

# 180° about Y — face the default model-viewer camera
_WIP_YAW_ROT = np.array(
    [
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)

DEFAULT_MAX_VOXELS = 80000
DEFAULT_CLAY_MAX_POINTS = 100000


def _as_numpy_coords(coords) -> np.ndarray:
    """Extract xyz int coords as (N, 3) float64 numpy."""
    if hasattr(coords, "detach"):
        arr = coords.detach().cpu().numpy()
    else:
        arr = np.asarray(coords)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected coords shape: {getattr(arr, 'shape', None)}")
    # [N, 4] = batch, x, y, z  OR already [N, 3]
    if arr.shape[1] >= 4:
        xyz = arr[:, 1:4].astype(np.float64)
    else:
        xyz = arr[:, :3].astype(np.float64)
    return xyz


def _subsample(xyz: np.ndarray, max_count: int) -> np.ndarray:
    if xyz.shape[0] <= max_count:
        return xyz
    rng = np.random.default_rng(0)
    idx = rng.choice(xyz.shape[0], size=max_count, replace=False)
    return xyz[idx]


def _apply_wip_transform(mesh) -> None:
    """Upright + front-facing for occupancy and clay WIP."""
    mesh.apply_transform(_PREVIEW_ROT)
    mesh.apply_transform(_WIP_UP_ROT)
    mesh.apply_transform(_WIP_YAW_ROT)


def _axis_rainbow_colors(centers: np.ndarray) -> np.ndarray:
    """RGB axis gradients: R←X, G←Y, B←Z over the cloud AABB (opaque)."""
    lo = centers.min(axis=0)
    span = np.maximum(centers.max(axis=0) - lo, 1e-8)
    t = (centers - lo) / span
    # Slight gamma so mid-range stays vivid instead of muddy gray
    t = np.clip(t, 0.0, 1.0) ** 0.85
    rgba = np.empty((centers.shape[0], 4), dtype=np.uint8)
    rgba[:, 0] = (t[:, 0] * 255.0).astype(np.uint8)
    rgba[:, 1] = (t[:, 1] * 255.0).astype(np.uint8)
    rgba[:, 2] = (t[:, 2] * 255.0).astype(np.uint8)
    rgba[:, 3] = 255
    return rgba


def coords_to_voxel_glb(
    coords,
    out_path: Union[str, Path],
    *,
    grid_resolution: Optional[int] = None,
    max_voxels: int = DEFAULT_MAX_VOXELS,
    color: tuple = (0.55, 0.72, 0.95, 1.0),
    rainbow: bool = True,
) -> dict[str, Any]:
    """
    Build a simple box-cloud GLB from occupancy coords.
    Centers the cloud in roughly [-0.5, 0.5]^3 like the final mesh AABB.
    By default colors voxels with XYZ→RGB axis gradients.
    """
    import trimesh
    from trimesh.voxel.ops import multibox

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xyz = _as_numpy_coords(coords)
    total = int(xyz.shape[0])
    xyz = _subsample(xyz, max_voxels)
    used = int(xyz.shape[0])

    if used == 0:
        mesh = trimesh.creation.box(extents=[0.05, 0.05, 0.05])
        mesh.visual.face_colors = [180, 180, 180, 255]
        _apply_wip_transform(mesh)
        mesh.export(str(out_path))
        return {"path": out_path, "voxels": 0, "total": 0}

    if grid_resolution is None:
        grid_resolution = int(max(xyz.max(), 1)) + 1
    grid_resolution = max(int(grid_resolution), 1)

    pitch = 0.92 / float(grid_resolution)
    centers = (xyz + 0.5) / float(grid_resolution) - 0.5
    if rainbow:
        colors = _axis_rainbow_colors(centers)
    else:
        rgba = np.array(
            [
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255),
                int(color[3] * 255),
            ],
            dtype=np.uint8,
        )
        colors = np.tile(rgba, (used, 1))

    mesh = multibox(centers, pitch=pitch, colors=colors)
    _apply_wip_transform(mesh)
    mesh.export(str(out_path))
    return {"path": out_path, "voxels": used, "total": total, "grid": grid_resolution}


def points_to_rainbow_glb(
    points,
    out_path: Union[str, Path],
    *,
    max_points: int = DEFAULT_CLAY_MAX_POINTS,
) -> dict[str, Any]:
    """Rainbow XYZ→RGB point cloud as tiny boxes (model-viewer-friendly)."""
    import trimesh
    from trimesh.voxel.ops import multibox

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(points, "detach"):
        pts = points.detach().float().cpu().numpy().astype(np.float64)
    else:
        pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Unexpected points shape: {getattr(pts, 'shape', None)}")
    pts = np.ascontiguousarray(pts[:, :3])
    total = int(pts.shape[0])
    pts = _subsample(pts, max_points)
    used = int(pts.shape[0])

    if used == 0:
        mesh = trimesh.creation.box(extents=[0.05, 0.05, 0.05])
        mesh.visual.face_colors = [180, 180, 180, 255]
        _apply_wip_transform(mesh)
        mesh.export(str(out_path))
        return {"path": out_path, "points": 0, "total": 0}

    extent = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
    # Box size scales with cloud so it reads as a solid cloud, not dust
    pitch = float(np.clip(extent / max(float(np.cbrt(used)) * 2.2, 1.0), 0.004, 0.02))
    colors = _axis_rainbow_colors(pts)
    mesh = multibox(pts, pitch=pitch, colors=colors)
    _apply_wip_transform(mesh)
    mesh.export(str(out_path))
    return {"path": out_path, "points": used, "total": total, "pitch": pitch}


def mesh_tensors_to_glb(
    vertices,
    faces=None,
    out_path: Union[str, Path] = None,
    color=(0.78, 0.78, 0.82, 1.0),
    *,
    max_faces: Optional[int] = None,
    max_points: Optional[int] = DEFAULT_CLAY_MAX_POINTS,
) -> dict[str, Any]:
    """Clay WIP: rainbow point cloud from mesh vertices (faces ignored)."""
    _ = (faces, color, max_faces)
    if out_path is None:
        raise ValueError("out_path is required")
    return points_to_rainbow_glb(
        vertices, out_path, max_points=int(max_points or DEFAULT_CLAY_MAX_POINTS)
    )


def shape_slat_to_clay_glb(
    pipeline,
    shape_slat,
    resolution: int,
    out_path: Union[str, Path],
    *,
    max_faces: Optional[int] = None,
    max_points: Optional[int] = DEFAULT_CLAY_MAX_POINTS,
) -> dict[str, Any]:
    """Decode shape SLat verts and export a rainbow point-cloud GLB."""
    _ = max_faces
    meshes, _subs = pipeline.decode_shape_slat(shape_slat, resolution)
    mesh = meshes[0]
    return points_to_rainbow_glb(
        mesh.vertices, out_path, max_points=int(max_points or DEFAULT_CLAY_MAX_POINTS)
    )
