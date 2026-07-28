"""
Pipeline initialization: Pixal3D-GGUF (default) or full HF safetensors.
"""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GGUF_DIR = ROOT / "models" / "Pixal3D-GGUF"
DEFAULT_DINO_DIR = ROOT / "models" / "dinov3-vitl16-pretrain-lvd1689m"


def _env_truthy(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _dino_model_name() -> str:
    override = os.environ.get("PIXAL3D_DINO_PATH", "").strip()
    if override:
        return override
    if DEFAULT_DINO_DIR.is_dir() and (DEFAULT_DINO_DIR / "config.json").is_file():
        return str(DEFAULT_DINO_DIR)
    return os.environ.get(
        "PIXAL3D_DINO_HF",
        "PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m",
    )


def _image_cond_configs() -> dict:
    dino = _dino_model_name()
    return {
        "ss": {
            "model_name": dino,
            "image_size": 512,
            "grid_resolution": 16,
        },
        "shape_512": {
            "model_name": dino,
            "image_size": 512,
            "grid_resolution": 32,
            "use_naf_upsample": True,
            "naf_target_size": 512,
        },
        "shape_1024": {
            "model_name": dino,
            "image_size": 1024,
            "grid_resolution": 64,
            "use_naf_upsample": True,
            "naf_target_size": 512,
        },
        "tex_1024": {
            "model_name": dino,
            "image_size": 1024,
            "grid_resolution": 64,
            "use_naf_upsample": True,
            "naf_target_size": 1024,
        },
    }


def _attach_image_cond_and_vram(pipeline, device: str, low_vram: bool):
    import torch
    from inference import build_image_cond_model

    configs = _image_cond_configs()
    print(f"[ImageCond] DINOv3 source: {configs['ss']['model_name']}")
    print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
    pipeline.image_cond_model_ss = build_image_cond_model(configs["ss"])
    pipeline.image_cond_model_shape_512 = build_image_cond_model(configs["shape_512"])
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(configs["shape_1024"])
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(configs["tex_1024"])

    if low_vram:
        print("[NAF] Pre-downloading NAF upsampler weights (CPU only)...")
        for attr in (
            "image_cond_model_ss",
            "image_cond_model_shape_512",
            "image_cond_model_shape_1024",
            "image_cond_model_tex_1024",
        ):
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, "use_naf_upsample", False):
                m._load_naf()
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
        print("[Pipeline] Low-VRAM mode enabled.")
    else:
        pipeline.low_vram = False
        pipeline.cuda()
        pipeline.image_cond_model_ss.cuda()
        pipeline.image_cond_model_shape_512.cuda()
        pipeline.image_cond_model_shape_1024.cuda()
        pipeline.image_cond_model_tex_1024.cuda()
        print("[NAF] Pre-loading NAF upsampler model...")
        for attr in (
            "image_cond_model_ss",
            "image_cond_model_shape_512",
            "image_cond_model_shape_1024",
            "image_cond_model_tex_1024",
        ):
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, "use_naf_upsample", False):
                m._load_naf()
        print("[Pipeline] Standard mode (all models on GPU).")
    return pipeline


def _finish_pipeline_like_from_pretrained(pipeline, args: dict):
    """Mirror Pixal3DImageTo3DPipeline.from_pretrained post-load wiring."""
    from pixal3d.pipelines import rembg, samplers

    pipeline.sparse_structure_sampler = getattr(samplers, args["sparse_structure_sampler"]["name"])(
        **args["sparse_structure_sampler"]["args"]
    )
    pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"]["params"]

    pipeline.shape_slat_sampler = getattr(samplers, args["shape_slat_sampler"]["name"])(
        **args["shape_slat_sampler"]["args"]
    )
    pipeline.shape_slat_sampler_params = args["shape_slat_sampler"]["params"]

    pipeline.tex_slat_sampler = getattr(samplers, args["tex_slat_sampler"]["name"])(
        **args["tex_slat_sampler"]["args"]
    )
    pipeline.tex_slat_sampler_params = args["tex_slat_sampler"]["params"]

    pipeline.shape_slat_normalization = args["shape_slat_normalization"]
    pipeline.tex_slat_normalization = args["tex_slat_normalization"]

    pipeline.image_cond_model_ss = None
    pipeline.image_cond_model_shape_512 = None
    pipeline.image_cond_model_shape_1024 = None
    pipeline.image_cond_model_tex_1024 = None

    rembg_cfg = args.get("rembg_model") or {"name": "BiRefNet", "args": {}}
    pipeline.rembg_model = getattr(rembg, rembg_cfg["name"])(**(rembg_cfg.get("args") or {}))

    pipeline.default_pipeline_type = args.get("default_pipeline_type", "1024_cascade")
    pipeline.pbr_attr_layout = {
        "base_color": slice(0, 3),
        "metallic": slice(3, 4),
        "roughness": slice(4, 5),
        "alpha": slice(5, 6),
    }
    pipeline._device = "cpu"
    pipeline._pretrained_args = args
    return pipeline


def _init_from_gguf(device: str, low_vram: bool):
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline
    from pixal_gguf.model_load import load_pipeline_models, normalize_quant

    root = Path(os.environ.get("PIXAL3D_MODEL_DIR", str(DEFAULT_GGUF_DIR)))
    quant = normalize_quant(os.environ.get("PIXAL3D_GGUF_QUANT", "Q5_K_M"))
    if not (root / "pipeline.json").is_file():
        raise FileNotFoundError(
            f"Pixal3D-GGUF bundle not found at {root}. "
            f"Run setup.bat / scripts/download_gguf_models.py first, "
            f"or set PIXAL3D_USE_GGUF=0 to use full HF weights."
        )

    print(f"[Pipeline] Loading Pixal3D-GGUF from {root} (quant={quant})...")
    models, args = load_pipeline_models(root, quant=quant)
    pipeline = Pixal3DImageTo3DPipeline(models)
    pipeline = _finish_pipeline_like_from_pretrained(pipeline, args)
    return _attach_image_cond_and_vram(pipeline, device, low_vram)


def _init_from_hf(device: str, low_vram: bool, model_path: Optional[str] = None):
    """Full safetensors path (escape hatch)."""
    from inference import IMAGE_COND_CONFIGS
    import inference as inf

    dino = _dino_model_name()
    patched = deepcopy(IMAGE_COND_CONFIGS)
    for cfg in patched.values():
        cfg["model_name"] = dino
    orig = inf.IMAGE_COND_CONFIGS
    inf.IMAGE_COND_CONFIGS = patched
    try:
        path = model_path or os.environ.get("PIXAL3D_HF_MODEL", "TencentARC/Pixal3D")
        print(f"[Pipeline] Loading full HF weights from {path}...")
        return inf.init_pipeline(model_path=path, device=device, low_vram=low_vram)
    finally:
        inf.IMAGE_COND_CONFIGS = orig


def weights_summary() -> str:
    """Human-readable weight source for job logs."""
    use_gguf = _env_truthy("PIXAL3D_USE_GGUF", "1")
    if use_gguf:
        root = Path(os.environ.get("PIXAL3D_MODEL_DIR", str(DEFAULT_GGUF_DIR)))
        from pixal_gguf.model_load import normalize_quant

        quant = normalize_quant(os.environ.get("PIXAL3D_GGUF_QUANT", "Q5_K_M"))
        if (root / "pipeline.json").is_file():
            return f"GGUF {quant} @ {root}"
        return f"GGUF requested but missing at {root} (will fall back to HF)"
    path = os.environ.get("PIXAL3D_HF_MODEL", "TencentARC/Pixal3D")
    return f"HF safetensors @ {path}"


def init_pipeline(device: str = "cuda", low_vram: bool = False):
    """
    Preferred entry for the web UI.
    Default: local Pixal3D-GGUF. Set PIXAL3D_USE_GGUF=0 for full HF.
    """
    use_gguf = _env_truthy("PIXAL3D_USE_GGUF", "1")
    if use_gguf:
        try:
            return _init_from_gguf(device, low_vram)
        except FileNotFoundError as e:
            print(f"[Pipeline] GGUF unavailable ({e}); falling back to HF safetensors.")
            return _init_from_hf(device, low_vram)
    return _init_from_hf(device, low_vram)
