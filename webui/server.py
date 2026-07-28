"""
Local Pixal3D web UI — FastAPI wrapper around Pixal3D/inference.py.
"""
from __future__ import annotations

import asyncio
import gc
import json
import math
import os
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths & env (must be set before importing Pixal3D / CUDA stacks)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
PIXAL_ROOT = ROOT / "Pixal3D"
OUTPUTS = ROOT / "outputs"
STATIC = Path(__file__).resolve().parent / "static"

# Cap input before rembg / MoGe so huge uploads cannot inflate VRAM.
# 1024² pixels + max side 1024 matches upstream preprocess intent.
MAX_INPUT_PIXELS = int(os.environ.get("PIXAL3D_MAX_INPUT_PIXELS", str(1024 * 1024)))
MAX_INPUT_SIDE = int(os.environ.get("PIXAL3D_MAX_INPUT_SIDE", "1024"))

sys.path.insert(0, str(PIXAL_ROOT))

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512",
)
# Prefer flash_attn when installed; callers can override via launch_gui.bat
if "ATTN_BACKEND" not in os.environ:
    try:
        import importlib.util

        os.environ["ATTN_BACKEND"] = (
            "flash_attn"
            if importlib.util.find_spec("flash_attn")
            else "sdpa"
        )
    except Exception:
        os.environ["ATTN_BACKEND"] = "sdpa"

os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(PIXAL_ROOT / "autotune_cache.json"),
)
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")

OUTPUTS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Pixal3D Web UI")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    "preview": {
        "low_vram": True,
        "resolution": 1024,
        "steps": 8,
        "max_tokens": 16384,
        "texture_size": 1024,
        "decimation": 200000,
    },
    "balanced": {
        "low_vram": True,
        "resolution": 1024,
        "steps": 12,
        "max_tokens": 32768,
        "texture_size": 2048,
        "decimation": 500000,
    },
    "max": {
        "low_vram": False,
        "resolution": 1536,
        "steps": 12,
        "max_tokens": 49152,
        "texture_size": 4096,
        "decimation": 1000000,
    },
}

DEFAULT_PRESET = os.environ.get("PIXAL3D_DEFAULT_PRESET", "balanced").strip().lower()
if DEFAULT_PRESET not in PRESETS:
    DEFAULT_PRESET = "balanced"

# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_pipeline = None
_pipeline_lock = threading.Lock()
_pipeline_low_vram: Optional[bool] = None
_gen_lock = threading.Lock()


def _job_update(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        job = _jobs.setdefault(
            job_id,
            {
                "id": job_id,
                "status": "queued",
                "progress": 0,
                "stage": "queued",
                "logs": [],
                "glb_url": None,
                "preview_url": None,
                "preview_label": None,
                "preview_history": [],
                "error": None,
            },
        )
        logs = kwargs.pop("log", None)
        job.update(kwargs)
        if logs is not None:
            job["logs"].append(str(logs))
            if len(job["logs"]) > 500:
                job["logs"] = job["logs"][-400:]


def _vram_snapshot() -> Optional[dict[str, float]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_b, total_b = torch.cuda.mem_get_info(0)
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        gb = 1024**3
        return {
            "total_gb": round(total_b / gb, 2),
            "free_gb": round(free_b / gb, 2),
            "used_gb": round((total_b - free_b) / gb, 2),
            "allocated_gb": round(allocated / gb, 2),
            "reserved_gb": round(reserved / gb, 2),
        }
    except Exception:
        return None


def _log(
    job_id: str,
    msg: str,
    progress: Optional[int] = None,
    stage: Optional[str] = None,
    *,
    with_vram: bool = False,
) -> None:
    if with_vram:
        snap = _vram_snapshot()
        if snap:
            msg = (
                f"{msg} | VRAM alloc={snap['allocated_gb']:.1f}G "
                f"reserved={snap['reserved_gb']:.1f}G "
                f"used={snap['used_gb']:.1f}/{snap['total_gb']:.1f}G"
            )
    print(f"[{job_id[:8]}] {msg}", flush=True)
    payload: dict[str, Any] = {"log": msg}
    if progress is not None:
        payload["progress"] = progress
    if stage is not None:
        payload["stage"] = stage
    _job_update(job_id, **payload)


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _flush_cuda(job_id: Optional[str] = None, label: str = "CUDA cache flushed") -> None:
    """Release orphaned tensors and return unused cached blocks to the driver."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()
    if job_id:
        _log(job_id, label, with_vram=True)


def _clamp_input_image(img, max_pixels: int = MAX_INPUT_PIXELS, max_side: int = MAX_INPUT_SIDE):
    """Downscale keeping aspect ratio so w*h <= max_pixels and max(w,h) <= max_side."""
    from PIL import Image

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in (img.mode or "") else "RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    scale = 1.0
    pixels = w * h
    if pixels > max_pixels:
        scale = min(scale, math.sqrt(max_pixels / float(pixels)))
    long_side = max(w, h) * scale
    if long_side > max_side:
        scale = min(scale, max_side / float(max(w, h)))
    if scale >= 0.999:
        return img
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def _emit_wip_preview(
    job_id: str,
    out_path: Path,
    label: str,
    *,
    progress: Optional[int] = None,
    stage: Optional[str] = None,
    detail: str = "",
) -> None:
    """Publish a WIP GLB URL; append to history so the UI can dwell on each stage."""
    rel = out_path.relative_to(OUTPUTS).as_posix()
    bust = uuid.uuid4().hex[:8]
    url = f"/outputs/{rel}?t={bust}"
    msg = f"WIP preview: {label}"
    if detail:
        msg = f"{msg} ({detail})"
    _log(job_id, msg, progress, stage, with_vram=True)
    with _jobs_lock:
        job = _jobs.setdefault(
            job_id,
            {
                "id": job_id,
                "status": "queued",
                "progress": 0,
                "stage": "queued",
                "logs": [],
                "glb_url": None,
                "preview_url": None,
                "preview_label": None,
                "preview_history": [],
                "error": None,
            },
        )
        history = job.setdefault("preview_history", [])
        entry = {"url": url, "label": label, "seq": len(history)}
        history.append(entry)
        job["preview_url"] = url
        job["preview_label"] = label


def _park_pipeline_on_cpu(pipeline) -> None:
    """Best-effort: park rembg / cond / flow models on CPU between low-VRAM jobs."""
    if pipeline is None:
        return
    # In standard (non-low-VRAM) mode models stay resident on GPU by design.
    if not getattr(pipeline, "low_vram", False):
        try:
            rembg = getattr(pipeline, "rembg_model", None)
            if rembg is not None and hasattr(rembg, "cpu"):
                rembg.cpu()
        except Exception:
            pass
        return
    try:
        rembg = getattr(pipeline, "rembg_model", None)
        if rembg is not None and hasattr(rembg, "cpu"):
            rembg.cpu()
    except Exception:
        pass
    for attr in (
        "image_cond_model_ss",
        "image_cond_model_shape_512",
        "image_cond_model_shape_1024",
        "image_cond_model_tex_1024",
        "image_cond_model",
    ):
        try:
            m = getattr(pipeline, attr, None)
            if m is not None and hasattr(m, "cpu"):
                m.cpu()
        except Exception:
            pass
    try:
        models = getattr(pipeline, "models", None)
        if isinstance(models, dict):
            for m in models.values():
                if m is not None and hasattr(m, "cpu"):
                    m.cpu()
    except Exception:
        pass


def _resolve_settings(
    preset: str,
    low_vram: bool,
    resolution: int,
    steps: int,
    max_tokens: int,
    texture_size: int,
    decimation: int,
) -> dict[str, Any]:
    base = dict(PRESETS.get(preset, PRESETS["balanced"]))
    # Explicit form values override preset when provided (>0 / meaningful)
    base["low_vram"] = low_vram
    if resolution in (1024, 1536):
        base["resolution"] = resolution
    else:
        # Auto / unset → always 1024; 1536 is an explicit UI choice
        base["resolution"] = 1024
    if steps > 0:
        base["steps"] = steps
    if max_tokens > 0:
        base["max_tokens"] = max_tokens
    if texture_size > 0:
        base["texture_size"] = texture_size
    if decimation > 0:
        base["decimation"] = decimation
    return base


# ---------------------------------------------------------------------------
# Pipeline helpers (mirrors Pixal3D/inference.py)
# ---------------------------------------------------------------------------

def _ensure_pipeline(low_vram: bool):
    global _pipeline, _pipeline_low_vram
    with _pipeline_lock:
        if _pipeline is not None and _pipeline_low_vram == low_vram:
            return _pipeline
        if _pipeline is not None:
            del _pipeline
            _pipeline = None
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

        from pipeline_init import init_pipeline

        _pipeline = init_pipeline(low_vram=low_vram)
        _pipeline_low_vram = low_vram
        return _pipeline


def _run_generation(
    job_id: str,
    image_path: Path,
    output_path: Path,
    seed: int,
    low_vram: bool,
    resolution: int,
    manual_fov: float,
    steps: int,
    max_tokens: int,
    texture_size: int,
    decimation: int,
) -> None:
    mesh_list = None
    shape_slat = None
    tex_slat = None
    mesh = None
    glb = None
    try:
        import numpy as np
        import torch
        from PIL import Image
        import o_voxel
        from inference import (
            distance_from_fov,
            get_camera_params_wild_moge,
            load_moge_model,
        )

        _flush_cuda(job_id, "Pre-run VRAM flush")
        _job_update(job_id, status="running", progress=2, stage="init")
        try:
            from pipeline_init import weights_summary

            _log(job_id, f"Weights: {weights_summary()}", 2, "init")
        except Exception:
            pass
        _log(
            job_id,
            f"Settings: low_vram={low_vram} res={resolution} steps={steps} "
            f"tokens={max_tokens} tex={texture_size} decim={decimation}",
            3,
            "init",
            with_vram=True,
        )
        _log(job_id, f"Loading pipeline (low_vram={low_vram})...", 5, "init", with_vram=True)
        pipeline = _ensure_pipeline(low_vram)

        _log(job_id, f"Preprocessing image: {image_path.name}", 10, "preprocess")
        img = Image.open(image_path)
        orig_w, orig_h = img.size
        img = _clamp_input_image(img)
        clamp_w, clamp_h = img.size
        if (clamp_w, clamp_h) != (orig_w, orig_h):
            _log(
                job_id,
                f"Input clamped {orig_w}x{orig_h} → {clamp_w}x{clamp_h} "
                f"(max_pixels={MAX_INPUT_PIXELS}, max_side={MAX_INPUT_SIDE})",
            )
        else:
            _log(job_id, f"Input size {orig_w}x{orig_h} (within clamp)")

        # Persist clamped original so rembg/MoGe never see the huge upload
        clamped_path = output_path.parent / f"input_clamped_{job_id[:8]}.png"
        img.save(clamped_path)

        image_preprocessed = pipeline.preprocess_image(img)
        # rembg may have been on GPU even outside low_vram; park it
        try:
            if getattr(pipeline, "rembg_model", None) is not None:
                pipeline.rembg_model.cpu()
        except Exception:
            pass
        _flush_cuda(job_id, "Post-preprocess flush")

        tmp_path = output_path.parent / f"_tmp_preprocessed_{job_id[:8]}.png"
        image_preprocessed.save(tmp_path)

        mesh_scale = 1.0
        extend_pixel = 0
        image_resolution = 512

        if manual_fov > 0:
            camera_angle_x = float(manual_fov)
            grid_point = torch.tensor([-1.0, 0.0, 0.0])
            distance = distance_from_fov(
                camera_angle_x,
                grid_point,
                torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
                mesh_scale,
                image_resolution,
            )["distance_from_x"]
            camera_params = {
                "camera_angle_x": camera_angle_x,
                "distance": distance,
                "mesh_scale": mesh_scale,
            }
            _log(
                job_id,
                f"Manual FOV: {math.degrees(manual_fov):.2f} deg, distance={distance:.4f}",
                20,
                "camera",
            )
        else:
            _log(job_id, "Loading MoGe-2 for camera estimation...", 15, "camera")
            moge_model = load_moge_model(device="cuda")
            _log(job_id, "Estimating camera parameters...", 20, "camera")
            camera_params = get_camera_params_wild_moge(
                str(tmp_path),
                moge_model,
                device="cuda",
                mesh_scale=mesh_scale,
                extend_pixel=extend_pixel,
                image_resolution=image_resolution,
            )
            _log(
                job_id,
                f"camera_angle_x={camera_params['camera_angle_x']:.4f}, "
                f"distance={camera_params['distance']:.4f}",
                25,
                "camera",
            )
            moge_model.cpu()
            del moge_model
            _flush_cuda(job_id, "Post-MoGe flush")

        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

        res = resolution if resolution in (1024, 1536) else 1024
        pipeline_type = f"{res}_cascade"
        _log(
            job_id,
            f"Running staged 3D pipeline ({pipeline_type})...",
            30,
            "generate",
            with_vram=True,
        )
        torch.manual_seed(seed)

        from preview_export import coords_to_voxel_glb, points_to_rainbow_glb
        from pixal3d.modules.sparse import SparseTensor

        ss_sampler_override = {
            "steps": steps,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.7,
            "rescale_t": 5.0,
        }
        shape_sampler_override = {
            "steps": steps,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
        }
        tex_sampler_override = {
            "steps": steps,
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
        }

        job_dir = output_path.parent
        job_dir.mkdir(parents=True, exist_ok=True)
        camera_angle_x = camera_params["camera_angle_x"]
        distance = camera_params["distance"]
        mesh_scale = camera_params.get("mesh_scale", 1.0)
        hr_resolution = res
        image = image_preprocessed

        # ---- Stage 1: Sparse structure ----
        _log(job_id, "Stage: sparse structure…", 35, "sparse", with_vram=True)
        cond_ss = pipeline.get_proj_cond_ss(
            [image],
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
        )
        ss_res = 32
        coords = pipeline.sample_sparse_structure(
            cond_ss, ss_res, 1, ss_sampler_override
        )
        del cond_ss
        try:
            preview_ss = job_dir / "preview_ss.glb"
            info = coords_to_voxel_glb(coords, preview_ss, grid_resolution=ss_res)
            _emit_wip_preview(
                job_id,
                preview_ss,
                "Sparse voxels",
                progress=42,
                stage="sparse",
                detail=f"{info.get('voxels', 0):,} voxels",
            )
        except Exception as preview_err:
            _log(job_id, f"WIP sparse preview skipped: {preview_err}")
        _flush_cuda(job_id, "Post-sparse flush")

        # ---- Stage 2: Shape LR 512 ----
        _log(job_id, "Stage: shape LR (512)…", 48, "shape_lr", with_vram=True)
        cond_shape_lr = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_shape_512,
            [image],
            coords,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
        )
        lr_slat = pipeline.sample_shape_slat(
            cond_shape_lr,
            pipeline.models["shape_slat_flow_model_512"],
            coords,
            shape_sampler_override,
        )
        del cond_shape_lr, coords
        _flush_cuda(job_id, "Post-shape-LR flush")

        # ---- Stage 3a: Upsample → denser voxels ----
        _log(job_id, "Stage: upsample occupancy…", 55, "upsample", with_vram=True)
        if pipeline.low_vram:
            pipeline.models["shape_slat_decoder"].to(pipeline.device)
            pipeline.models["shape_slat_decoder"].low_vram = True
        hr_coords = pipeline.models["shape_slat_decoder"].upsample(lr_slat, upsample_times=4)
        if pipeline.low_vram:
            pipeline.models["shape_slat_decoder"].cpu()
            pipeline.models["shape_slat_decoder"].low_vram = False

        lr_resolution = 512
        actual_hr_resolution = hr_resolution
        while True:
            grid_res_tok = actual_hr_resolution // 16
            quant_coords = torch.cat(
                [
                    hr_coords[:, :1],
                    ((hr_coords[:, 1:] + 0.5) / lr_resolution * (grid_res_tok - 1))
                    .round()
                    .int(),
                ],
                dim=1,
            )
            hr_coords_unique = quant_coords.unique(dim=0)
            num_tokens = hr_coords_unique.shape[0]
            if num_tokens < max_tokens or actual_hr_resolution == 1024:
                if actual_hr_resolution != hr_resolution:
                    _log(
                        job_id,
                        f"Token cap: resolution reduced to {actual_hr_resolution}",
                    )
                break
            actual_hr_resolution -= 128

        actual_grid_res = actual_hr_resolution // 16
        # Park upsample leftovers before CPU voxel export (avoids ~10G reserved spike)
        del lr_slat, hr_coords, quant_coords
        _flush_cuda(job_id, "Post-upsample flush")
        try:
            preview_up = job_dir / "preview_up.glb"
            info = coords_to_voxel_glb(
                hr_coords_unique,
                preview_up,
                grid_resolution=actual_grid_res,
                color=(0.45, 0.82, 0.72, 1.0),
            )
            _emit_wip_preview(
                job_id,
                preview_up,
                "Dense voxels",
                progress=60,
                stage="upsample",
                detail=f"{info.get('voxels', 0):,} / {info.get('total', 0):,} voxels",
            )
        except Exception as preview_err:
            _log(job_id, f"WIP dense preview skipped: {preview_err}")

        # ---- Stage 3b: Shape HR ----
        _log(
            job_id,
            f"Stage: shape HR ({actual_hr_resolution})…",
            65,
            "shape_hr",
            with_vram=True,
        )
        cond_shape_hr = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_shape_1024,
            [image],
            hr_coords_unique,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            grid_resolution_override=actual_grid_res,
        )
        noise_hr = SparseTensor(
            feats=torch.randn(
                hr_coords_unique.shape[0],
                pipeline.models["shape_slat_flow_model_1024"].in_channels,
            ).to(pipeline.device),
            coords=hr_coords_unique,
        )
        sampler_params_hr = {
            **pipeline.shape_slat_sampler_params,
            **shape_sampler_override,
        }
        flow_model_hr = pipeline.models["shape_slat_flow_model_1024"]
        if pipeline.low_vram:
            flow_model_hr.to(pipeline.device)
        hr_slat = pipeline.shape_slat_sampler.sample(
            flow_model_hr,
            noise_hr,
            **cond_shape_hr,
            **sampler_params_hr,
            verbose=True,
            tqdm_desc=f"Sampling HR shape SLat (proj, {actual_hr_resolution})",
        ).samples
        if pipeline.low_vram:
            flow_model_hr.cpu()
        std = torch.tensor(pipeline.shape_slat_normalization["std"])[None].to(
            hr_slat.device
        )
        mean = torch.tensor(pipeline.shape_slat_normalization["mean"])[None].to(
            hr_slat.device
        )
        shape_slat = hr_slat * std + mean
        del cond_shape_hr, noise_hr, hr_slat, hr_coords_unique
        _flush_cuda(job_id, "Post-shape-HR flush")

        # Cheap 3rd WIP: HR occupancy voxels (no extra shape decode — avoids 24GB spike)
        try:
            preview_shape = job_dir / "preview_shape.glb"
            info = coords_to_voxel_glb(
                shape_slat.coords,
                preview_shape,
                grid_resolution=actual_grid_res,
                color=(0.78, 0.78, 0.82, 1.0),
            )
            _emit_wip_preview(
                job_id,
                preview_shape,
                "Shape occupancy",
                progress=72,
                stage="shape_hr",
                detail=f"{info.get('voxels', 0):,} voxels",
            )
        except Exception as preview_err:
            _log(job_id, f"WIP shape occupancy preview skipped: {preview_err}")

        # ---- Stage 4: Texture ----
        _log(job_id, "Stage: texture…", 78, "texture", with_vram=True)
        tex_grid_res = actual_hr_resolution // 16
        cond_tex = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_tex_1024,
            [image],
            shape_slat.coords,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            grid_resolution_override=tex_grid_res,
        )
        tex_slat = pipeline.sample_tex_slat(
            cond_tex,
            pipeline.models["tex_slat_flow_model_1024"],
            shape_slat,
            tex_sampler_override,
        )
        del cond_tex
        _flush_cuda(job_id, "Post-texture flush")

        # ---- Stage 5: Decode (use pipeline.decode_latent for @torch.no_grad) ----
        _log(job_id, "Stage: decode latent…", 85, "decode", with_vram=True)
        grid_res = actual_hr_resolution
        with torch.no_grad():
            mesh_list = pipeline.decode_latent(shape_slat, tex_slat, grid_res)
        mesh = mesh_list[0]
        # Clay WIP: rainbow point cloud from decode verts (mesh preview was unreliable)
        try:
            n_verts = int(mesh.vertices.shape[0]) if hasattr(mesh.vertices, "shape") else -1
            _log(
                job_id,
                f"Exporting clay point cloud ({n_verts:,} verts → capped)…",
                86,
                "clay",
                with_vram=True,
            )
            preview_clay = job_dir / "preview_clay.glb"
            info = points_to_rainbow_glb(
                mesh.vertices.detach() if hasattr(mesh.vertices, "detach") else mesh.vertices,
                preview_clay,
            )
            detail = f"{info.get('points', 0):,} pts"
            if info.get("total") and info["total"] != info.get("points"):
                detail = f"{info['points']:,} of {info['total']:,} pts"
            _emit_wip_preview(
                job_id,
                preview_clay,
                "Clay points",
                progress=87,
                stage="clay",
                detail=detail,
            )
        except Exception as clay_err:
            _log(job_id, f"WIP clay export skipped: {clay_err}")
        del shape_slat, tex_slat
        shape_slat = None
        tex_slat = None
        _flush_cuda(job_id, "Post-pipeline flush")
        _log(job_id, "Pipeline finished. Extracting GLB...", 88, "export", with_vram=True)

        # Detach before o_voxel — requires_grad tensors break .numpy()
        verts = mesh.vertices.detach() if hasattr(mesh.vertices, "detach") else mesh.vertices
        faces = mesh.faces.detach() if hasattr(mesh.faces, "detach") else mesh.faces
        attrs = mesh.attrs.detach() if hasattr(mesh.attrs, "detach") else mesh.attrs
        coords = mesh.coords.detach() if hasattr(mesh.coords, "detach") else mesh.coords

        glb = o_voxel.postprocess.to_glb(
            vertices=verts,
            faces=faces,
            attr_volume=attrs,
            coords=coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=grid_res,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=True,
        )

        # Match model-viewer: PREVIEW_ROT then 180° Y (front-facing; up already correct)
        rot = np.array(
            [
                [-1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        yaw = np.array(
            [
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        glb.apply_transform(rot)
        glb.apply_transform(yaw)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        glb.export(str(output_path), extension_webp=True)

        del mesh, mesh_list, glb
        mesh = None
        mesh_list = None
        glb = None
        _park_pipeline_on_cpu(pipeline)
        _flush_cuda(job_id, "Post-export VRAM flush")

        rel = output_path.relative_to(OUTPUTS).as_posix()
        glb_url = f"/outputs/{rel}"
        _log(job_id, f"Done. GLB saved: {output_path.name}", 100, "done", with_vram=True)
        _job_update(
            job_id,
            status="done",
            progress=100,
            stage="done",
            glb_url=glb_url,
            preview_url=glb_url,
            preview_label="Final",
        )

    except Exception as e:
        tb = traceback.format_exc()
        _log(job_id, f"ERROR: {e}")
        _log(job_id, tb)
        _job_update(job_id, status="error", stage="error", error=str(e))
    finally:
        # Always drop large refs + flush so repeated runs don't climb reserved VRAM
        mesh_list = None
        shape_slat = None
        tex_slat = None
        mesh = None
        glb = None
        try:
            _park_pipeline_on_cpu(_pipeline)
        except Exception:
            pass
        _flush_cuda(job_id, "Between-run VRAM flush")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/presets")
async def list_presets():
    return {"default": DEFAULT_PRESET, "presets": PRESETS}


@app.get("/api/health")
async def health():
    import importlib.util

    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu = {
                "name": torch.cuda.get_device_name(0),
                "vram_gb": round(props.total_memory / 1024**3, 1),
            }
    except Exception:
        pass
    return {
        "ok": True,
        "attn_backend": os.environ.get("ATTN_BACKEND"),
        "gpu": gpu,
        "vram": _vram_snapshot(),
        "pipeline_loaded": _pipeline is not None,
        "pipeline_low_vram": _pipeline_low_vram,
        "default_preset": DEFAULT_PRESET,
        "flash_attn": importlib.util.find_spec("flash_attn") is not None,
    }


@app.get("/api/vram")
async def vram():
    snap = _vram_snapshot()
    if snap is None:
        raise HTTPException(503, "CUDA not available")
    return snap


@app.post("/api/generate")
async def generate(
    image: UploadFile = File(...),
    seed: int = Form(42),
    low_vram: str = Form("true"),
    resolution: int = Form(1024),
    fov: float = Form(-1.0),
    preset: str = Form("balanced"),
    steps: int = Form(0),
    max_tokens: int = Form(0),
    texture_size: int = Form(0),
    decimation: int = Form(0),
):
    if not _gen_lock.acquire(blocking=False):
        raise HTTPException(409, "A generation is already running. Wait for it to finish.")

    use_low_vram = _as_bool(low_vram)
    preset_key = (preset or DEFAULT_PRESET).strip().lower()
    if preset_key not in PRESETS:
        preset_key = DEFAULT_PRESET

    settings = _resolve_settings(
        preset=preset_key,
        low_vram=use_low_vram,
        resolution=resolution,
        steps=steps,
        max_tokens=max_tokens,
        texture_size=texture_size,
        decimation=decimation,
    )

    job_id = uuid.uuid4().hex
    job_dir = OUTPUTS / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(image.filename or "input.png").suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        suffix = ".png"
    image_path = job_dir / f"input{suffix}"
    data = await image.read()
    image_path.write_bytes(data)

    output_path = job_dir / "output.glb"
    _job_update(job_id, status="queued", progress=0, stage="queued")
    _log(
        job_id,
        f"Queued. preset={preset_key} seed={seed} "
        f"low_vram={settings['low_vram']} resolution={settings['resolution']} "
        f"steps={settings['steps']} tokens={settings['max_tokens']} "
        f"tex={settings['texture_size']} decim={settings['decimation']}",
    )

    def _worker():
        try:
            _run_generation(
                job_id=job_id,
                image_path=image_path,
                output_path=output_path,
                seed=seed,
                low_vram=bool(settings["low_vram"]),
                resolution=int(settings["resolution"]),
                manual_fov=fov,
                steps=int(settings["steps"]),
                max_tokens=int(settings["max_tokens"]),
                texture_size=int(settings["texture_size"]),
                decimation=int(settings["decimation"]),
            )
        finally:
            _gen_lock.release()

    threading.Thread(target=_worker, daemon=True).start()
    return {"job_id": job_id, "settings": settings}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return dict(job)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    async def event_stream():
        last_log = 0
        while True:
            with _jobs_lock:
                job = _jobs.get(job_id)
                if not job:
                    yield f"data: {json.dumps({'error': 'not found'})}\n\n"
                    return
                logs = job["logs"][last_log:]
                last_log = len(job["logs"])
                payload = {
                    "status": job["status"],
                    "progress": job["progress"],
                    "stage": job["stage"],
                    "logs": logs,
                    "glb_url": job.get("glb_url"),
                    "preview_url": job.get("preview_url"),
                    "preview_label": job.get("preview_label"),
                    "preview_history": list(job.get("preview_history") or []),
                    "error": job.get("error"),
                    "vram": _vram_snapshot(),
                }
            yield f"data: {json.dumps(payload)}\n\n"
            if job["status"] in ("done", "error"):
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def main():
    import logging

    import uvicorn

    class _QuietVramAccessFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
            except Exception:
                return True
            return "GET /api/vram" not in msg

    logging.getLogger("uvicorn.access").addFilter(_QuietVramAccessFilter())

    host = os.environ.get("PIXAL3D_HOST", "127.0.0.1")
    port = int(os.environ.get("PIXAL3D_PORT", "7860"))
    print(f"Pixal3D Web UI -> http://{host}:{port}")
    print(f"ATTN_BACKEND={os.environ.get('ATTN_BACKEND')}")
    print(f"DEFAULT_PRESET={DEFAULT_PRESET}")
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    use_gguf = os.environ.get("PIXAL3D_USE_GGUF", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    print(f"PIXAL3D_USE_GGUF={int(use_gguf)} QUANT={os.environ.get('PIXAL3D_GGUF_QUANT', 'Q5_K_M')}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
