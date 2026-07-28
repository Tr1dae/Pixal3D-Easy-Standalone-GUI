@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo  Pixal3D Windows setup
echo  Stack: Python 3.12 + PyTorch 2.8 + CUDA 12.8 + Pixal3D-GGUF
echo ============================================================
echo.

where py >nul 2>&1
if %ERRORLEVEL%==0 (
  set "PYLAUNCH=py -3.12"
) else (
  where python >nul 2>&1
  if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.12 and retry.
    exit /b 1
  )
  set "PYLAUNCH=python"
)

echo [0/8] Ensuring Pixal3D source ...
if not exist "Pixal3D\pixal3d" (
  echo   Cloning https://github.com/TencentARC/Pixal3D ...
  git clone --branch master --depth 1 https://github.com/TencentARC/Pixal3D.git Pixal3D
  if errorlevel 1 (
    echo ERROR: failed to clone Pixal3D
    exit /b 1
  )
) else (
  echo   Pixal3D/ already present
)

echo [1/8] Creating venv ...
if not exist "venv\Scripts\python.exe" (
  %PYLAUNCH% -m venv venv
  if errorlevel 1 (
    echo ERROR: failed to create venv
    exit /b 1
  )
) else (
  echo   venv already exists
)

set "PY=venv\Scripts\python.exe"
set "PIP=venv\Scripts\python.exe -m pip"

echo [2/8] Upgrading pip / wheel / setuptools ...
%PIP% install --upgrade pip wheel setuptools
if errorlevel 1 exit /b 1

echo [3/8] Installing PyTorch 2.8 + cu128 ...
%PIP% install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 exit /b 1

echo [4/8] Installing Pixal3D Python dependencies ...
%PIP% install ninja pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless trimesh transformers zstandard kornia timm diffusers accelerate plyfile pandas tensorboard einops
if errorlevel 1 exit /b 1
%PIP% install "git+https://github.com/microsoft/MoGe.git"
if errorlevel 1 exit /b 1
%PIP% install "https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl"
if errorlevel 1 (
  echo WARN: utils3d wheel failed, trying git fallback ...
  %PIP% install "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
)
%PIP% install -r Pixal3D\requirements.txt
if errorlevel 1 exit /b 1

echo [5/8] Installing Windows CUDA wheels (cu128 / torch2.8 / cp312) ...
%PIP% install --no-deps ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel-latest/o_voxel-0.0.1%%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrec_render-latest/nvdiffrec_render-0.0.1%%2Bcu128torch2.8-cp312-cp312-win_amd64.whl"
if errorlevel 1 (
  echo ERROR: CUDA extension wheels failed to install
  exit /b 1
)

echo   Installing flash_attn wheel ...
%PIP% install --no-deps "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flash_attn-latest/flash_attn-2.8.3%%2Bcu128torch2.8-cp312-cp312-win_amd64.whl"
if errorlevel 1 (
  echo WARN: flash_attn wheel failed — GUI will default to ATTN_BACKEND=sdpa
)

echo   Installing triton-windows 3.4.x (recommended for torch 2.8) ...
%PIP% install "triton-windows==3.4.0.post21"
if errorlevel 1 (
  echo WARN: triton-windows 3.4.0.post21 failed, keeping whatever is installed
)

echo   Attempting natten Windows wheel (needed for NAF upsampling) ...
%PIP% install --no-deps "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/natten-latest/natten-0.21.6%%2Bcu128torch2.8-cp312-cp312-win_amd64.whl"
if errorlevel 1 (
  echo WARN: natten wheel failed — trying pip source build ...
  %PIP% install "natten==0.21.0" --no-build-isolation
  if errorlevel 1 echo WARN: natten not installed — NAF may fail; see README_SETUP.md
)

echo [6/8] Web UI deps + import aliases ...
%PIP% install -r requirements-webui.txt
if errorlevel 1 exit /b 1

%PY% scripts\alias_cuda_packages.py
if errorlevel 1 (
  echo WARN: some CUDA package aliases failed — continuing to env check
)

echo [7/8] Downloading Pixal3D-GGUF (Q5_K_M) + DINOv3 mirror ...
echo   This is a multi-GB download on first run; skips files already present.
set PYTHONPATH=%CD%\Pixal3D;%PYTHONPATH%
%PY% scripts\download_gguf_models.py --quant Q5_K_M
if errorlevel 1 (
  echo WARN: model download failed — GUI can still use PIXAL3D_USE_GGUF=0 full HF weights
)

echo [8/8] Environment check ...
set PYTHONPATH=%CD%\Pixal3D;%PYTHONPATH%
%PY% scripts\env_check.py
set "CHECK_ERR=!ERRORLEVEL!"

echo.
echo ============================================================
if not "!CHECK_ERR!"=="0" (
  echo  Setup finished with warnings/errors. See output above.
) else (
  echo  Setup complete.
)
echo  Models: models\Pixal3D-GGUF + models\dinov3-vitl16-pretrain-lvd1689m
echo  Next: double-click launch_gui.bat
echo ============================================================
endlocal
exit /b %CHECK_ERR%
