#!/bin/bash
set -e -x

mkdir -p logs/profiles_mirage
NSYS_ARGS="--trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --stop-on-exit=true --wait=primary"

for i in 128 256 512 1024 2048; do
    for j in 16 32 64 128 256 512 1024 2048; do
        # Skip if j > i
        if [ $j -gt $i ]; then
            continue
        fi
        nsys profile $NSYS_ARGS -o logs/profiles_mirage/prefill_global-1,32,32,$j,$i,128 \
            python3 -u scripts/run_mirage.py 1,32,$j,$i,128
        nsys profile $NSYS_ARGS -o logs/profiles_mirage/prefill_causal-1,32,32,$j,$i,128 \
            python3 -u scripts/run_mirage.py 1,32,$j,$i,128 --mask
    done
done
