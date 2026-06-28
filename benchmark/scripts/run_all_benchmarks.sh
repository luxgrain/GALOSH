#!/bin/bash
# Run all v5 benchmarks sequentially
# v5 final: BM3D-CFA (s=1.0), BM3D-CFA (s=0.5), RAW L/C BM3D, DnCNN-B, DRUNet
# All images saved for all methods

cd "$HOME/denoise_eval"

echo "=========================================="
echo " Kodak (24 images x 6 ISOs)"
echo "=========================================="
python scripts/run_benchmark_v5.py --dataset kodak 2>&1

echo ""
echo "=========================================="
echo " CBSD68 (68 images x 6 ISOs)"
echo "=========================================="
python scripts/run_benchmark_v5.py --dataset cbsd68 2>&1

echo ""
echo "=========================================="
echo " McMaster (18 images x 6 ISOs)"
echo "=========================================="
python scripts/run_benchmark_v5.py --dataset mcmaster 2>&1

echo ""
echo "ALL BENCHMARKS COMPLETE"
date
