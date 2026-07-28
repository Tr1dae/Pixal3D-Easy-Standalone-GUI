@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo venv missing. Run setup.bat first.
  pause
  exit /b 1
)

if "%~1"=="" (
  echo Usage: run_inference.bat path\to\image.png [output.glb] [--low_vram] [--resolution 1024]
  echo Example: run_inference.bat Pixal3D\assets\images\0_img.png outputs\sample.glb --low_vram
  pause
  exit /b 1
)

set "PYTHONPATH=%CD%\Pixal3D;%PYTHONPATH%"
set "OPENCV_IO_ENABLE_OPENEXR=1"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
if not defined ATTN_BACKEND set "ATTN_BACKEND=flash_attn"
set "FLEX_GEMM_AUTOTUNE_CACHE_PATH=%CD%\Pixal3D\autotune_cache.json"
set "FLEX_GEMM_AUTOTUNER_VERBOSE=1"

set "IMG=%~1"
set "OUT=%~2"
if "%OUT%"=="" set "OUT=outputs\cli_output.glb"
if "%OUT:~0,2%"=="--" (
  set "OUT=outputs\cli_output.glb"
  set "EXTRA=%~2 %~3 %~4 %~5 %~6"
) else (
  set "EXTRA=%~3 %~4 %~5 %~6 %~7"
)

if not exist "outputs" mkdir outputs

echo Image: %IMG%
echo Output: %OUT%
venv\Scripts\python.exe Pixal3D\inference.py --image "%IMG%" --output "%OUT%" %EXTRA%
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" (
  echo Inference failed with code %ERR%
  pause
)
endlocal
exit /b %ERR%
