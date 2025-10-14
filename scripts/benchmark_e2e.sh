#!/bin/bash
set -e -x

export CUDA_VISIBLE_DEVICES=1
mkdir -p logs/e2e_profiles

for seq_len in 8192 4096 2048 1024; do
    python3 scripts/benchmark_e2e.py llama 1 $seq_len --profiler nsys
    python3 scripts/benchmark_e2e.py llama 1 $seq_len --profiler nsys --decode
done
for seq_len in 1024 4096; do
    python3 scripts/benchmark_e2e.py gemma2 1 $seq_len --profiler nsys
    python3 scripts/benchmark_e2e.py gemma2 1 $seq_len --profiler nsys --decode
done
for seq_len in 8192 4096 2048 1024; do
    python3 scripts/benchmark_e2e.py vit 1 $seq_len --profiler nsys
done
