# ============================================================================
# GALOSH — standalone reference build            (code: Apache-2.0)
# ----------------------------------------------------------------------------
# The standalone CLI / exe is the CANONICAL reference implementation used for
# the paper and benchmark reproduction (darktable is a future/separate demo).
#
# Toolchain: a C99 compiler + OpenMP; GPU targets additionally need OpenCL
# (headers + libOpenCL). On Windows we build with the MSYS2 ucrt64 gcc, which
# ships the OpenCL headers/lib:
#     export PATH=/c/msys64/ucrt64/bin:$PATH
#     make all            # CPU reference builds (RAW + YUV)
#     make gpu            # GPU streaming builds (need OpenCL + the *.cl at run time)
#
# Portable flags by default. For a local speed build:
#     make CFLAGS="-O3 -march=native -funroll-loops -fopenmp"
# ============================================================================

CC      ?= gcc
CFLAGS  ?= -O3 -fopenmp -std=gnu11
LM       = -lm
LCL      = -lOpenCL

RAW_CPU      = galosh_raw_cpu.exe          # GALOSH-RAW  FP32 reference (canonical)
RAW_CPU_INT  = galosh_raw_cpu_int.exe      # GALOSH-RAW  INT32 r32 fixed-point reference
RAW_GPU      = galosh_raw_gpu.exe          # GALOSH-RAW  GPU (o32 FP32 / i16 INT16 streaming)
YUV_CPU      = galosh_yuv_cpu.exe          # GALOSH-YUV/RGB  FP32 reference (canonical)
YUV_GPU      = galosh_yuv_gpu.exe          # GALOSH-YUV/RGB  GPU

.PHONY: all raw raw-gpu yuv yuv-gpu gpu test bench-small check-no-int64 clean help

all: raw yuv                               ## canonical CPU reference builds (RAW + YUV)

# ---- GALOSH-RAW (canonical) ------------------------------------------------
raw: $(RAW_CPU) $(RAW_CPU_INT)             ## RAW FP32 + INT32 r32 CPU references
$(RAW_CPU):     galosh_raw_cpu.c galosh_cpu.h
	$(CC) $(CFLAGS) $< -o $@ $(LM)
$(RAW_CPU_INT): galosh_raw_cpu_int.c galosh_cpu_int.h
	$(CC) $(CFLAGS) $< -o $@ $(LM)
raw-gpu: $(RAW_GPU)                         ## RAW GPU (o32/i16); needs OpenCL + galosh*.cl at run time
$(RAW_GPU):     galosh_raw_gpu.c galosh_gpu.h
	$(CC) $(CFLAGS) $< -o $@ $(LCL) $(LM)

# ---- GALOSH-YUV/RGB (canonical) --------------------------------------------
yuv: $(YUV_CPU)                            ## YUV/RGB FP32 CPU reference
$(YUV_CPU):     galosh_yuv_cpu.c
	$(CC) $(CFLAGS) $< -o $@ $(LM)
yuv-gpu: $(YUV_GPU)                         ## YUV/RGB GPU; needs OpenCL + galosh.cl at run time
$(YUV_GPU):     galosh_yuv_gpu.c
	$(CC) $(CFLAGS) $< -o $@ $(LCL) $(LM)

gpu: raw-gpu yuv-gpu                        ## all GPU streaming builds

# ---- harness ---------------------------------------------------------------
test: all                                  ## smoke tests (constant/random/odd/small/high-noise/near-black, RAW+YUV)
	bash tests/run_smoke.sh
bench-small: all                           ## small reproducible benchmark -> CSV/JSON
	bash tests/run_bench_small.sh
check-no-int64:                            ## fail if the INT fixed-point shipping path uses a native 64-bit type
	bash check_no_int64.sh

clean:
	rm -f $(RAW_CPU) $(RAW_CPU_INT) $(RAW_GPU) $(YUV_CPU) $(YUV_GPU)

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n",$$1,$$2}'
