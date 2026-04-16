@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 2>/dev/null
if not defined INCLUDE (
    echo ERROR: vcvarsall failed, INCLUDE not set
    exit /b 1
)
cd /d "C:\Users\luxgrain\denoise_eval\standalone"
cl.exe /nologo /c /O2 /openmp yuv_galosh_core.c /Fo:yuv_galosh_core_msvc.obj
if errorlevel 1 ( echo C_COMPILE_FAILED & exit /b 1 )
cl.exe /nologo /LD /O2 /openmp /EHsc galosh_avisynth.cpp yuv_galosh_core_msvc.obj /I. /Fe:GALOSHDenoise_msvc.dll
if errorlevel 1 ( echo LINK_FAILED & exit /b 1 )
echo BUILD_MSVC_OK
