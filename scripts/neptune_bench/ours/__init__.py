from .attn_gqa import (
    NeptuneGQADecodeRunner,
    NeptuneGQARunner,
)
from .attn_plain import (
    NeptuneAttentionRunner,
    NeptuneDecodeRunner,
    _schedule_attention_flash,
    causal_mask_cond,
    create_alibi_attention,
    create_general_attention,
    schedule_full_attn_flash,
)


def tir_softcap(softcap: float = 10.0):
    from tvm.tir import tanh

    return lambda score, *_: tanh(score / softcap) * softcap


def tir_windowed_mask_cond(left_window_size: int):
    from tvm import tir

    return lambda b, h0, h1, q_idx, kv_idx: tir.all(
        q_idx >= kv_idx, q_idx - kv_idx < left_window_size
    )
