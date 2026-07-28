# Pixal3D Easy Standalone GUI

Windows-native FastAPI web UI for [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) — image upload, progress/logs, live VRAM meter, and GLB 3D preview. One-click `.bat` installers and launchers.

## Requirements

- Windows 10/11
- NVIDIA GPU (24 GB VRAM recommended; use **Balanced** / low-VRAM on 16–24 GB)
- Python 3.12
- CUDA-capable drivers (stack targets PyTorch 2.8 + CUDA 12.8)

## Quick start

1. Clone this repo.
2. Double-click **`setup.bat`** (clones Pixal3D if needed, creates `venv`, installs CUDA wheels, downloads Pixal3D-GGUF + DINOv3 — can take a while).
3. Double-click **`launch_gui.bat`** — opens http://127.0.0.1:7860 (GGUF `Q5_K_M` by default).
4. Drop an image → leave **Balanced** preset → Generate.

CLI:

```bat
run_inference.bat Pixal3D\assets\images\0_img.png outputs\sample.glb --low_vram --resolution 1024
```

## VRAM presets

| Preset | Typical peak | Notes |
|--------|--------------|-------|
| **Preview** | ~10–12 GB | Fast checks |
| **Balanced (default)** | ~12–17 GB | Daily driver on 24 GB cards |
| **Max quality** | 24 GB+ | Can spill into shared GPU memory and thrash — use only with clear VRAM headroom |

## Layout

```
setup.bat                 one-time install (+ clones TencentARC/Pixal3D, downloads GGUF)
launch_gui.bat            start web UI (GGUF + Balanced defaults)
launch_gui_lowvram.bat    force Balanced preset env
run_inference.bat         CLI wrapper
webui/                    FastAPI + static UI + GGUF loader
scripts/                  env check, CUDA aliases, GGUF download
models/                   Pixal3D-GGUF + DINOv3 (created by setup, not in git)
Pixal3D/                  upstream (created by setup.bat, not in git)
venv/                     created by setup.bat
outputs/                  generation results
```

## Notes

- Default weights are quantized GGUF (~6 GB). Full HF safetensors: `set PIXAL3D_USE_GGUF=0`.
- Upstream Pixal3D is MIT; this wrapper is also intended for MIT use.
- ComfyUI / GGUF install guides are a different stack and are not used here.

More detail: see [README_SETUP.md](README_SETUP.md).
