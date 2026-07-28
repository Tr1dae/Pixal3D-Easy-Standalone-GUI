"""Quick environment check for Pixal3D Windows setup."""
from __future__ import annotations

import importlib.util
import sys


def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    print("=" * 60)
    print("Pixal3D environment check")
    print("=" * 60)
    print(f"Python: {sys.version}")

    errors = []

    try:
        import torch

        print(f"torch: {torch.__version__}  cuda={torch.version.cuda}")
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"capability: {torch.cuda.get_device_capability(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            errors.append("CUDA not available in torch")
    except Exception as e:
        errors.append(f"torch import failed: {e}")

    checks = {
        "flex_gemm": ("flex_gemm", "flex_gemm_ap"),
        "cumesh": ("cumesh", "cumesh_vb"),
        "o_voxel": ("o_voxel", "o_voxel_vb_ap", "o_voxel_vb"),
        "o_voxel.postprocess": ("o_voxel.postprocess",),
        "drtk": ("drtk",),
        "nvdiffrast": ("nvdiffrast",),
        "flash_attn": ("flash_attn", "flash_attn_interface"),
        "moge": ("moge",),
        "utils3d": ("utils3d",),
        "fastapi": ("fastapi",),
        "pixal3d": ("pixal3d",),
    }

    for label, names in checks.items():
        found = next((n for n in names if _has(n)), None)
        if found:
            print(f"[OK] {label}: {found}")
        else:
            print(f"[MISSING] {label}: tried {names}")
            if label in ("flex_gemm", "cumesh", "o_voxel", "pixal3d"):
                errors.append(f"missing {label}")

    # Attention backend note
    if not _has("flash_attn") and not _has("flash_attn_interface"):
        print("[WARN] No flash_attn — set ATTN_BACKEND=sdpa")

    # NATTEN / NAF
    if _has("natten"):
        try:
            import natten

            has_lib = getattr(natten, "HAS_LIBNATTEN", None)
            print(f"[OK] natten: {natten.__version__} HAS_LIBNATTEN={has_lib}")
            if not has_lib:
                print("[WARN] NATTEN without libnatten — NAF quality may fall back / fail")
        except Exception as e:
            print(f"[WARN] natten import error: {e}")
    else:
        print("[WARN] natten not installed — NAF upsampling may fail on first generate")

    print("=" * 60)
    if errors:
        print("RESULT: FAIL")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
