# Pixal3D local setup (Windows)

RTX 4090 / Python 3.12 / PyTorch 2.8 + CUDA 12.8 stack with a small local web UI.

## Quick start

1. Double-click **`setup.bat`** (first time only; can take a long time — includes GGUF + DINOv3 downloads).
2. Double-click **`launch_gui.bat`** (or **`launch_gui_lowvram.bat`**) — browser opens at http://127.0.0.1:7860
3. Drop an image → leave **Balanced** preset → Generate → watch progress / VRAM / console → orbit the GLB when done.

CLI without UI:

```bat
run_inference.bat Pixal3D\assets\images\0_img.png outputs\sample.glb --low_vram --resolution 1024
```

## Weights (GGUF by default)

The GUI loads quantized flow DiTs from **`models/Pixal3D-GGUF`** ([duharlequin/Pixal3D-GGUF](https://huggingface.co/duharlequin/Pixal3D-GGUF)), default quant **`Q5_K_M`**, plus fp16 decoders. DINOv3 comes from a local mirror of **PIA-SPACE-LAB** (ungated) under `models/dinov3-vitl16-pretrain-lvd1689m`.

| Env | Meaning |
|-----|---------|
| `PIXAL3D_USE_GGUF=1` (default) | Local GGUF bundle |
| `PIXAL3D_USE_GGUF=0` | Full `TencentARC/Pixal3D` safetensors from Hugging Face (~24 GB) |
| `PIXAL3D_GGUF_QUANT` | e.g. `Q5_K_M` (default), `Q8_0`, `Q4_K_M` |
| `PIXAL3D_MODEL_DIR` | Override GGUF root |
| `PIXAL3D_DINO_PATH` | Override local DINOv3 folder |

Re-download / change quant:

```bat
venv\Scripts\python.exe scripts\download_gguf_models.py --quant Q8_0
```

Disk for the default GGUF stack is roughly **~6 GB** (flows + decoders) vs **~24 GB** full HF.

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

Live VRAM readout sits in the header (allocated / used). Yellow/red warning when used > ~92% of dedicated memory. `/api/vram` poll lines are filtered from the uvicorn access log.

## What got installed

| Piece | Role |
|-------|------|
| `Pixal3D/` | Upstream [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) (Trellis.2 backbone) |
| `venv/` | Python 3.12 virtualenv |
| Windows CUDA wheels | `flex_gemm_ap`, `cumesh_vb`, full `o_voxel`, `drtk`, `nvdiffrast`, `nvdiffrec_render`, `flash_attn`, `natten` (Pozzetti builds) |
| `webui/` | FastAPI + model-viewer GUI + GGUF loader |
| `models/Pixal3D-GGUF` | Quantized flow DiTs + fp16 decoders |
| `outputs/` | Uploaded jobs + GLBs |

Import aliases map Windows wheel names (`flex_gemm_ap`, `cumesh_vb`) to what Pixal3D imports (`flex_gemm`, `cumesh`). The full `o_voxel` wheel (with `postprocess.to_glb`) is installed directly.

## Notes

- Launch sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512` to reduce allocator fragmentation.
- If `flash_attn` failed during setup, set before launch:
  ```bat
  set ATTN_BACKEND=sdpa
  launch_gui.bat
  ```
- Official TRELLIS.2 install is Linux-oriented; this repo uses the Windows wheel path instead of compiling CUDA extensions.
- [trellis2cpp](https://github.com/rms80/trellis2cpp) is **not** used (stage-1 geometry only; not full Pixal3D).
- ComfyUI-Trellis2-GGUF is a separate stack; this app keeps Pixal3D but reuses the same GGUF-on-the-fly dequant approach for lighter VRAM.

## Layout

```
setup.bat                 one-time install + GGUF download
launch_gui.bat            start web UI (Balanced + GGUF defaults)
launch_gui_lowvram.bat    same, forces PIXAL3D_DEFAULT_PRESET=balanced
run_inference.bat         CLI wrapper
webui/server.py           FastAPI app
webui/pipeline_init.py    GGUF / HF pipeline factory
webui/pixal_gguf/         GGUF loader (City96-style)
scripts/download_gguf_models.py
models/                   GGUF + DINOv3 (gitignored)
Pixal3D/                  cloned upstream
venv/                     virtualenv
outputs/                  results
```

## Re-check environment

```bat
venv\Scripts\python.exe scripts\env_check.py
```
