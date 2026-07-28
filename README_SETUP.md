# Pixal3D local setup (Windows)

RTX 4090 / Python 3.12 / PyTorch 2.8 + CUDA 12.8 stack with a small local web UI.

## Quick start

1. Double-click **`setup.bat`** (first time only; can take a long time).
2. Double-click **`launch_gui.bat`** (or **`launch_gui_lowvram.bat`**) — browser opens at http://127.0.0.1:7860
3. Drop an image → leave **Balanced** preset → Generate → watch progress / VRAM / console → orbit the GLB when done.

CLI without UI:

```bat
run_inference.bat Pixal3D\assets\images\0_img.png outputs\sample.glb --low_vram --resolution 1024
```

## VRAM presets (use these)

| Preset | Typical peak | Notes |
|--------|--------------|-------|
| **Preview** | ~10–12 GB | Fast checks; lower steps/tokens/texture |
| **Balanced (default)** | ~12–17 GB | Daily driver on 24 GB cards; low-VRAM + 1024 |
| **Max quality** | 24 GB+ | All models resident + 1536 + 4K textures — can spill into shared GPU memory and thrash |

**Do not run Max** unless Task Manager shows plenty of free dedicated VRAM and nothing else is using the GPU. A full 1536 run with `low_vram=false` can fill 24 GB dedicated and spill another ~20 GB into shared (system) memory — wall clock then stretches past 10 minutes even though the GPU shows 100%.

UI defaults:

- Low VRAM **on**
- Resolution **1024**
- Texture **2048**, tokens **32768**, decimation **500k**

Live VRAM readout sits in the header (allocated / used). Yellow/red warning when used &gt; ~92% of dedicated memory.

## What got installed

| Piece | Role |
|-------|------|
| `Pixal3D/` | Upstream [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) (Trellis.2 backbone) |
| `venv/` | Python 3.12 virtualenv |
| Windows CUDA wheels | `flex_gemm_ap`, `cumesh_vb`, full `o_voxel`, `drtk`, `nvdiffrast`, `nvdiffrec_render`, `flash_attn`, `natten` (Pozzetti builds) |
| `webui/` | FastAPI + model-viewer GUI |
| `outputs/` | Uploaded jobs + GLBs |

Import aliases map Windows wheel names (`flex_gemm_ap`, `cumesh_vb`) to what Pixal3D imports (`flex_gemm`, `cumesh`). The full `o_voxel` wheel (with `postprocess.to_glb`) is installed directly.

## Notes

- **First run downloads multi-GB Hugging Face weights** (`TencentARC/Pixal3D`, DINOv3, MoGe, NAF). Needs disk space and network.
- Launch sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512` to reduce allocator fragmentation.
- If `flash_attn` failed during setup, set before launch:
  ```bat
  set ATTN_BACKEND=sdpa
  launch_gui.bat
  ```
- Official TRELLIS.2 install is Linux-oriented; this repo uses the Windows wheel path instead of compiling CUDA extensions.
- [trellis2cpp](https://github.com/rms80/trellis2cpp) is **not** used (stage-1 geometry only; not full Pixal3D).
- **ComfyUI / GGUF guides** (e.g. [PixelArtistry Trellis2 install](https://pixel-artistry.com/Trellis2InstallationGuide)) target ComfyUI-Trellis2 quantized loaders. That stack is separate from this FastAPI + upstream Pixal3D Python pipeline; we reuse the *ideas* (low VRAM, smaller resolution/texture) but do not install GGUF here.

## Layout

```
setup.bat                 one-time install
launch_gui.bat            start web UI (Balanced defaults)
launch_gui_lowvram.bat    same, forces PIXAL3D_DEFAULT_PRESET=balanced
run_inference.bat         CLI wrapper
webui/server.py           FastAPI app
Pixal3D/                  cloned upstream
venv/                     virtualenv
outputs/                  results
```

## Re-check environment

```bat
venv\Scripts\python.exe scripts\env_check.py
```
