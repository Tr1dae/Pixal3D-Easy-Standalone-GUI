"""
Download Pixal3D-GGUF (Q5_K_M flows + fp16 decoders) and a local DINOv3 mirror.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

ROOT = Path(__file__).resolve().parent.parent
GGUF_REPO = "duharlequin/Pixal3D-GGUF"
DINO_REPO = "PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m"

DEFAULT_QUANT = "Q5_K_M"

# Flow basenames (no folder / extension)
FLOW_BASES = [
    "ss_flow_img_dit_1_3B_64_bf16",
    "slat_flow_img2shape_dit_1_3B_512_bf16",
    "slat_flow_img2shape_dit_1_3B_1024_bf16",
    "slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
]

DECODER_BASES = [
    "ss_dec_conv3d_16l8_fp16",
    "shape_dec_next_dc_f16c32_fp16",
    "tex_dec_next_dc_f16c32_fp16",
]

FOLDER_FOR = {
    "ss_flow_": "Sparse",
    "slat_flow_img2shape_": "shape",
    "slat_flow_imgshape2tex_": "texture",
    "ss_dec_": "decoder",
    "shape_dec_": "decoder",
    "tex_dec_": "decoder",
}


def _folder(basename: str) -> str:
    for prefix, folder in FOLDER_FOR.items():
        if basename.startswith(prefix):
            return folder
    raise ValueError(f"Unknown basename: {basename}")


def _download(repo: str, remote: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  exists: {dest.relative_to(ROOT)}")
        return dest
    print(f"  download: {repo}/{remote}")
    cached = hf_hub_download(repo_id=repo, filename=remote)
    shutil.copy2(cached, dest)
    print(f"  -> {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def download_gguf(out_dir: Path, quant: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Pixal3D-GGUF → {out_dir} (quant={quant}) ===")

    # pipeline.json
    _download(GGUF_REPO, "pipeline.json", out_dir / "pipeline.json")

    # Discover available remote files once (helps when layout drifts)
    try:
        remote_files = set(list_repo_files(GGUF_REPO))
    except Exception as e:
        print(f"WARN: could not list repo files ({e}); using expected paths")
        remote_files = set()

    for base in FLOW_BASES:
        folder = _folder(base)
        dest_dir = out_dir / folder
        json_remote = f"{folder}/{base}.json"
        gguf_remote = f"{folder}/{base}_{quant}.gguf"
        if remote_files and json_remote not in remote_files:
            # try without folder prefix in listing edge cases
            print(f"WARN: {json_remote} not in repo listing")
        _download(GGUF_REPO, json_remote, dest_dir / f"{base}.json")
        _download(GGUF_REPO, gguf_remote, dest_dir / f"{base}_{quant}.gguf")

    for base in DECODER_BASES:
        folder = _folder(base)
        dest_dir = out_dir / folder
        _download(GGUF_REPO, f"{folder}/{base}.json", dest_dir / f"{base}.json")
        _download(
            GGUF_REPO,
            f"{folder}/{base}.safetensors",
            dest_dir / f"{base}.safetensors",
        )

    print("GGUF bundle ready.")


def download_dino(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== DINOv3 mirror → {out_dir} ===")
    for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
        _download(DINO_REPO, name, out_dir / name)
    print("DINOv3 mirror ready.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quant",
        default=DEFAULT_QUANT,
        help=f"GGUF quant suffix (default {DEFAULT_QUANT})",
    )
    ap.add_argument(
        "--gguf-dir",
        type=Path,
        default=ROOT / "models" / "Pixal3D-GGUF",
    )
    ap.add_argument(
        "--dino-dir",
        type=Path,
        default=ROOT / "models" / "dinov3-vitl16-pretrain-lvd1689m",
    )
    ap.add_argument("--skip-gguf", action="store_true")
    ap.add_argument("--skip-dino", action="store_true")
    args = ap.parse_args()

    try:
        if not args.skip_gguf:
            download_gguf(args.gguf_dir, args.quant)
        if not args.skip_dino:
            download_dino(args.dino_dir)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print("\nAll requested downloads finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
