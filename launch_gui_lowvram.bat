@echo off
REM Convenience launcher that forces the Balanced / low-VRAM defaults.
setlocal
cd /d "%~dp0"
set "PIXAL3D_DEFAULT_PRESET=balanced"
call "%~dp0launch_gui.bat"
endlocal
