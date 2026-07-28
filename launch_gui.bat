@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo venv missing. Run setup.bat first.
  pause
  exit /b 1
)

if not exist "Pixal3D\pixal3d" (
  echo Pixal3D source missing. Clone failed or incomplete.
  pause
  exit /b 1
)

set "PYTHONPATH=%CD%\Pixal3D;%PYTHONPATH%"
set "OPENCV_IO_ENABLE_OPENEXR=1"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512"
if not defined ATTN_BACKEND set "ATTN_BACKEND=flash_attn"
set "FLEX_GEMM_AUTOTUNE_CACHE_PATH=%CD%\Pixal3D\autotune_cache.json"
set "FLEX_GEMM_AUTOTUNER_VERBOSE=1"
if not defined PIXAL3D_DEFAULT_PRESET set "PIXAL3D_DEFAULT_PRESET=balanced"
if not defined PIXAL3D_USE_GGUF set "PIXAL3D_USE_GGUF=1"
if not defined PIXAL3D_GGUF_QUANT set "PIXAL3D_GGUF_QUANT=Q5_K_M"
if not defined PIXAL3D_MODEL_DIR set "PIXAL3D_MODEL_DIR=%CD%\models\Pixal3D-GGUF"
if not defined PIXAL3D_DINO_PATH if exist "%CD%\models\dinov3-vitl16-pretrain-lvd1689m\config.json" set "PIXAL3D_DINO_PATH=%CD%\models\dinov3-vitl16-pretrain-lvd1689m"
set "PIXAL3D_HOST=127.0.0.1"
set "PIXAL3D_PORT=7860"

echo Starting Pixal3D Web UI on http://%PIXAL3D_HOST%:%PIXAL3D_PORT%
echo ATTN_BACKEND=%ATTN_BACKEND%
echo DEFAULT_PRESET=%PIXAL3D_DEFAULT_PRESET%
echo PIXAL3D_USE_GGUF=%PIXAL3D_USE_GGUF% QUANT=%PIXAL3D_GGUF_QUANT%
echo PYTORCH_CUDA_ALLOC_CONF=%PYTORCH_CUDA_ALLOC_CONF%
start "" "http://%PIXAL3D_HOST%:%PIXAL3D_PORT%/"

venv\Scripts\python.exe webui\server.py
if errorlevel 1 (
  echo.
  echo Server exited with an error.
  pause
)
endlocal
