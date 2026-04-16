@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >/dev/null 2>&1
cd /d "C:\Users\luxgrain\denoise_eval\standalone"

cl.exe /c /O2 /openmp /nologo yuv_galosh_core.c /Fo:yuv_galosh_core_msvc.obj
if %ERRORLEVEL% neq 0 ( echo C_COMPILE_FAILED & goto end )

cl.exe /LD /O2 /openmp /nologo /EHsc galosh_avisynth.cpp yuv_galosh_core_msvc.obj /I. /Fe:GALOSHDenoise_msvc.dll
if %ERRORLEVEL% neq 0 ( echo LINK_FAILED & goto end )

echo BUILD_MSVC_OK
:end
