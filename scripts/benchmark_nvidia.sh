#!/bin/bash
set -e -x

export CUDA_VISIBLE_DEVICES=1
mkdir -p logs/profiles

for op in prefill_global prefill_causal prefill_gqa prefill_alibi prefill_softcap prefill_windowed \
    decode_causal decode_gqa decode_alibi decode_softcap; do
    for batch in 1 2 4 8 16 32; do
        python3 -u -m scripts.felix_attn profile $op $batch,8192 --profiler nsys --repeat 15
    done
done
