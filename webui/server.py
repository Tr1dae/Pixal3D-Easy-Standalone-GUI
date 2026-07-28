"""
Local Pixal3D web UI — FastAPI wrapper around Pixal3D/inference.py.
"""
from __future__ import annotations

import asyncio
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

        from inference import init_pipeline

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

        _job_update(job_id, status="running", progress=2, stage="init")
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
        image_preprocessed = pipeline.preprocess_image(img)

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
            torch.cuda.empty_cache()

        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

        res = resolution if resolution in (1024, 1536) else 1024
        pipeline_type = f"{res}_cascade"
        _log(
            job_id,
            f"Running 3D pipeline ({pipeline_type})...",
            30,
            "generate",
            with_vram=True,
        )
        torch.manual_seed(seed)

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

        _log(
            job_id,
            "Stage: sparse structure → shape → texture...",
            40,
            "generate",
            with_vram=True,
        )
        mesh_list, (shape_slat, tex_slat, grid_res) = pipeline.run(
            image_preprocessed,
            camera_params=camera_params,
            seed=seed,
            sparse_structure_sampler_params=ss_sampler_override,
            shape_slat_sampler_params=shape_sampler_override,
            tex_slat_sampler_params=tex_sampler_override,
            preprocess_image=False,
            return_latent=True,
            pipeline_type=pipeline_type,
            max_num_tokens=max_tokens,
        )
        mesh = mesh_list[0]
        torch.cuda.empty_cache()
        _log(job_id, "Pipeline finished. Extracting GLB...", 85, "export", with_vram=True)

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
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

        rot = np.array(
            [
                [-1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        glb.apply_transform(rot)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        glb.export(str(output_path), extension_webp=True)
        torch.cuda.empty_cache()

        rel = output_path.relative_to(OUTPUTS).as_posix()
        glb_url = f"/outputs/{rel}"
        _log(job_id, f"Done. GLB saved: {output_path.name}", 100, "done", with_vram=True)
        _job_update(job_id, status="done", progress=100, stage="done", glb_url=glb_url)

    except Exception as e:
        tb = traceback.format_exc()
        _log(job_id, f"ERROR: {e}")
        _log(job_id, tb)
        _job_update(job_id, status="error", stage="error", error=str(e))
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


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
                    "error": job.get("error"),
                    "vram": _vram_snapshot(),
                }
            yield f"data: {json.dumps(payload)}\n\n"
            if job["status"] in ("done", "error"):
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def main():
    import uvicorn

    host = os.environ.get("PIXAL3D_HOST", "127.0.0.1")
    port = int(os.environ.get("PIXAL3D_PORT", "7860"))
    print(f"Pixal3D Web UI -> http://{host}:{port}")
    print(f"ATTN_BACKEND={os.environ.get('ATTN_BACKEND')}")
    print(f"DEFAULT_PRESET={DEFAULT_PRESET}")
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
