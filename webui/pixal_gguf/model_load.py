"""
Local path resolution + model construction for Pixal3D-GGUF weights.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger("Pixal3D-GGUF")

# duharlequin/Pixal3D-GGUF folder layout
_FOLDER_MAP = (
    ("ss_flow_", "Sparse"),
    ("slat_flow_img2shape_", "shape"),
    ("slat_flow_imgshape2tex_", "texture"),
    ("ss_dec_", "decoder"),
    ("shape_dec_", "decoder"),
    ("tex_dec_", "decoder"),
)

_FLOW_PREFIXES = ("ss_flow_", "slat_flow_img2shape_", "slat_flow_imgshape2tex_")


def normalize_quant(quant: str) -> str:
    q = (quant or "Q5_K_M").strip()
    if q.startswith("GGUF_"):
        q = q[5:]
    # Common aliases
    if q in ("Q4_K", "Q4"):
        q = "Q4_K_M"
    if q in ("Q5_K", "Q5"):
        q = "Q5_K_M"
    return q


def folder_for_basename(basename: str) -> str:
    for prefix, folder in _FOLDER_MAP:
        if basename.startswith(prefix):
            return folder
    return ""


def is_flow_model(basename: str) -> bool:
    return any(basename.startswith(p) for p in _FLOW_PREFIXES)


def resolve_model_files(
    root: Path,
    ckpt_rel: str,
    *,
    quant: str = "Q5_K_M",
    enable_gguf: bool = True,
) -> tuple[Path, Path, bool]:
    """
    Resolve (config.json, weights, is_gguf) under a Pixal3D-GGUF root.

    ``ckpt_rel`` is the pipeline.json value, e.g. ``ckpts/ss_flow_img_dit_1_3B_64_bf16``
    or just the basename.
    """
    root = Path(root)
    basename = Path(ckpt_rel).name
    folder = folder_for_basename(basename)
    base_dir = root / folder if folder else root

    config_path = base_dir / f"{basename}.json"
    if not config_path.is_file():
        # Some bundles keep json next to pipeline.json
        alt = root / f"{basename}.json"
        if alt.is_file():
            config_path = alt
        else:
            raise FileNotFoundError(f"Missing arch JSON for {basename}: tried {config_path}")

    quant = normalize_quant(quant)
    want_gguf = enable_gguf and is_flow_model(basename)
    if want_gguf:
        gguf_path = base_dir / f"{basename}_{quant}.gguf"
        if gguf_path.is_file():
            return config_path, gguf_path, True
        # Fall back to unquantized bf16.gguf if present
        bf16 = base_dir / f"{basename}.gguf"
        if bf16.is_file():
            return config_path, bf16, True
        raise FileNotFoundError(
            f"Missing GGUF for {basename} ({quant}): expected {gguf_path}"
        )

    st = base_dir / f"{basename}.safetensors"
    if st.is_file():
        return config_path, st, False
    raise FileNotFoundError(f"Missing safetensors for {basename}: expected {st}")


def _tensor_shape(t) -> tuple:
    ts = getattr(t, "tensor_shape", None)
    if ts is not None:
        return tuple(ts)
    return tuple(t.shape)


def _infer_arch_from_sd(sd: dict, config: dict) -> dict:
    """Patch config args from actual weight shapes (GGUF metadata can drift)."""
    args = dict(config.get("args") or {})

    for key in (
        "t_embedder.mlp.0.weight",
        "t_embedder.mlp.2.weight",
        "input_layer.weight",
        "input_proj.weight",
        "x_embedder.weight",
        "blocks.0.self_attn.to_qkv.weight",
    ):
        if key in sd:
            shape = _tensor_shape(sd[key])
            if "blocks.0.self_attn" in key:
                args["model_channels"] = shape[1]
            elif "input_layer" in key or "input_proj" in key or "x_embedder" in key:
                args["model_channels"] = shape[0]
            else:
                args["model_channels"] = shape[0]
            break

    for key in ("input_layer.weight", "input_proj.weight", "x_embedder.weight"):
        if key in sd:
            args["in_channels"] = _tensor_shape(sd[key])[1]
            break

    for key in ("output_layer.1.weight", "final_layer.linear.weight", "proj_out.weight", "out_layer.weight"):
        if key in sd:
            args["out_channels"] = _tensor_shape(sd[key])[0]
            break

    block_ids = {
        int(m.group(1))
        for k in sd
        for m in [re.match(r"^blocks\.(\d+)\.", k)]
        if m
    }
    if block_ids:
        args["num_blocks"] = max(block_ids) + 1

    mc = args.get("model_channels")
    if mc:
        for key in ("blocks.0.mlp.mlp.0.weight", "blocks.0.ff.net.0.proj.weight"):
            if key in sd:
                args["mlp_ratio"] = _tensor_shape(sd[key])[0] / float(mc)
                break

    config = dict(config)
    config["args"] = args
    return config


def _build_empty_model(config: dict, **kwargs):
    import pixal3d.models as models

    model_class = getattr(models, config["name"])
    merged = {**(config.get("args") or {}), **kwargs}
    sig = inspect.signature(model_class.__init__)
    has_var = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    final_args = {k: v for k, v in merged.items() if has_var or k in sig.parameters}

    # Skip expensive random init
    orig_init = getattr(model_class, "initialize_weights", None)
    if orig_init:
        model_class.initialize_weights = lambda self: None
    init = torch.nn.init
    names = [
        "normal_",
        "kaiming_uniform_",
        "uniform_",
        "zeros_",
        "ones_",
        "kaiming_normal_",
        "xavier_uniform_",
        "xavier_normal_",
        "constant_",
    ]
    orig = {n: getattr(init, n) for n in names if hasattr(init, n)}
    noop = lambda tensor, *a, **k: tensor
    for n in orig:
        setattr(init, n, noop)
    try:
        model = model_class(**final_args)
    finally:
        for n, fn in orig.items():
            setattr(init, n, fn)
        if orig_init:
            model_class.initialize_weights = orig_init
    return model


def _maybe_fix_rope(model) -> None:
    if not hasattr(model, "rope_phases") or model.rope_phases is None:
        return
    try:
        from pixal3d.modules.attention.rope import RotaryPositionEmbedder

        logger.info("Regenerating rope_phases for GGUF model…")
        pos_embedder = RotaryPositionEmbedder(model.model_channels // model.num_heads, 3)
        coords = torch.meshgrid(
            *[torch.arange(res, device=model.rope_phases.device) for res in [model.resolution] * 3],
            indexing="ij",
        )
        coords = torch.stack(coords, dim=-1).reshape(-1, 3)
        rope_phases = pos_embedder(coords)
        model.rope_phases.copy_(rope_phases)
    except Exception as e:
        logger.warning("Failed to regenerate rope_phases: %s", e)


def load_component(
    root: Path,
    ckpt_rel: str,
    *,
    quant: str = "Q5_K_M",
    enable_gguf: bool = True,
) -> torch.nn.Module:
    """Load one pipeline component (flow GGUF or decoder safetensors)."""
    from safetensors.torch import load_file

    from .gguf_utils import convert_to_ggml, load_gguf_checkpoint

    config_path, weights_path, is_gguf = resolve_model_files(
        root, ckpt_rel, quant=quant, enable_gguf=enable_gguf
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if is_gguf:
        logger.info("Loading GGUF %s", weights_path.name)
        sd, metadata = load_gguf_checkpoint(str(weights_path))
        if metadata:
            meta_map = {
                "trellis.attention.head_count": "num_heads",
                "trellis.model.model_channels": "model_channels",
                "trellis.model.num_blocks": "num_blocks",
                "trellis.model.in_channels": "in_channels",
                "trellis.model.out_channels": "out_channels",
            }
            args = config.setdefault("args", {})
            for k, v in meta_map.items():
                if k in metadata:
                    args[v] = metadata[k]
        config = _infer_arch_from_sd(sd, config)
        model = _build_empty_model(config)
        model = convert_to_ggml(model)
        model.load_state_dict(sd, strict=False)
        _maybe_fix_rope(model)
    else:
        logger.info("Loading safetensors %s", weights_path.name)
        model = _build_empty_model(config)
        model.load_state_dict(load_file(str(weights_path)), strict=False)

    model.eval()
    return model


def load_pipeline_models(
    root: Path,
    pipeline_json: Optional[Path] = None,
    *,
    quant: str = "Q5_K_M",
) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    """
    Load all models listed in pipeline.json.
    Returns (models_dict, pretrained_args).
    """
    root = Path(root)
    pipeline_json = Path(pipeline_json) if pipeline_json else root / "pipeline.json"
    if not pipeline_json.is_file():
        raise FileNotFoundError(f"pipeline.json not found: {pipeline_json}")

    with open(pipeline_json, "r", encoding="utf-8") as f:
        full = json.load(f)
    args = full.get("args") or full
    model_map = args["models"]

    models: dict[str, torch.nn.Module] = {}
    for key, rel in model_map.items():
        print(f"[Pixal3D-GGUF] Loading {key} ← {rel}", flush=True)
        models[key] = load_component(root, rel, quant=quant, enable_gguf=True)
    return models, args
