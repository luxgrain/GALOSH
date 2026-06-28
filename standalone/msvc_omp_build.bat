@echo off
set CL_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe
set MSVC_INC=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\include
set MSVC_LIB=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\lib\x64
set SDK_UCRT_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt
set SDK_UM_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um
set SDK_SH_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared
set SDK_UCRT_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64
set SDK_UM_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64
set WORK=%USERPROFILE%\denoise_eval\standalone
cd /d %WORK%

rem /openmp:llvm ? OpenMP 4.0+ ??
"%CL_PATH%" /nologo /c /O2 /openmp:llvm /TC /I"%MSVC_INC%" /I"%SDK_UCRT_INC%" /I"%SDK_UM_INC%" /I"%SDK_SH_INC%" yuv_galosh_core.c /Fo:yuv_galosh_core_omp.obj 2>&1
if errorlevel 1 ( echo OMP_LLVM_FAILED & goto try_exp )
echo OMP_LLVM_OK & goto link
:try_exp
"%CL_PATH%" /nologo /c /O2 /openmp:experimental /TC /I"%MSVC_INC%" /I"%SDK_UCRT_INC%" /I"%SDK_UM_INC%" /I"%SDK_SH_INC%" yuv_galosh_core.c /Fo:yuv_galosh_core_omp.obj 2>&1
if errorlevel 1 ( echo OMP_EXP_FAILED & exit /b 1 )
echo OMP_EXP_OK
:link
"%CL_PATH%" /nologo /LD /O2 /openmp:llvm /EHsc /I"%MSVC_INC%" /I"%SDK_UCRT_INC%" /I"%SDK_UM_INC%" /I"%SDK_SH_INC%" /I"%WORK%" galosh_avisynth.cpp yuv_galosh_core_omp.obj /Fe:GALOSHDenoise_omp.dll /link /LIBPATH:"%MSVC_LIB%" /LIBPATH:"%SDK_UCRT_LIB%" /LIBPATH:"%SDK_UM_LIB%"
if errorlevel 1 ( echo LINK_FAILED & exit /b 1 )
echo BUILD_OMP_OK