#!/bin/bash
set -e -x

mkdir -p logs/profiles

for op in prefill_global prefill_causal prefill_gqa prefill_alibi prefill_softcap prefill_windowed \
    decode_causal decode_gqa decode_alibi decode_softcap; do
    for seq_len in 256 512 1024 2048 4096 8192 16384 32768; do
        python3 -u -m scripts.felix_attn profile $op 1,$seq_len --profiler rocprof --repeat 15
    done
done
