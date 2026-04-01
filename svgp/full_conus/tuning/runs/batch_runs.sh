#!/bin/bash
for bs in 16384 32768 49152; do
    for workers_per_gpu in 1 2 3 4; do
        echo "=== Testing batch_size=$bs  workers_per_gpu=$workers_per_gpu ==="
        python -u loso_cv_svgp_parallel.py --n_gpus 1 --n_inducing 4096 --n_epochs 1 \
            --n_sites $workers_per_gpu --states MT --batch_size $bs --workers_per_gpu $workers_per_gpu \
            && echo "PASSED: batch_size=$bs workers_per_gpu=$workers_per_gpu" \
            || echo "FAILED: batch_size=$bs workers_per_gpu=$workers_per_gpu"
        echo ""
    done
done
