import flashinfer
import torch

KV_LAYOUT = "NHD"
WORKSPACE_BUFFER = torch.empty(512 * 1024 * 1024, dtype=torch.int8).to(0)


def attention_fp8_paged(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_scale, v_scale, is_causal: bool
):
    """Run Fused Attention (FlashAttention) with FP16 query and FP8 key/value.
    The input must have 4D shape [batch, n_head, seq_len, headdim],
    but the actual attention runs in "NHD" layout."""

    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(WORKSPACE_BUFFER, KV_LAYOUT)
    batch, qo_head, seq_len, headdim = q.shape
    kv_head = k.shape[1]
    assert k.shape == v.shape == (batch, kv_head, seq_len, headdim)
    assert (kv_dtype := k.dtype) == v.dtype == torch.float8_e5m2
    assert (q_dtype := q.dtype) == torch.float16

    # Layout for Q: [batch * seq_len, qo_head, headdim]
    q = q.cpu().transpose(1, 2).view(batch * seq_len, qo_head, headdim).contiguous()
    # Layout for K and V: [seq_len, batch, kv_head, headdim]
    k = k.cpu().permute(2, 0, 1, 3).contiguous()
    v = v.cpu().permute(2, 0, 1, 3).contiguous()

    page_size = 1
    q_indptr = torch.arange(0, batch + 1).int() * seq_len
    kv_indptr = torch.arange(0, batch + 1).int() * seq_len
    kv_indices = torch.arange(0, seq_len).int()
    kv_last_page_len = torch.full((batch,), (seq_len - 1) % page_size + 1, dtype=torch.int32)

    wrapper.plan(
        q_indptr.cuda(),
        kv_indptr.cuda(),
        kv_indices.cuda(),
        kv_last_page_len.cuda(),
        qo_head,
        kv_head,
        headdim,
        page_size,
        causal=is_causal,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
        use_fp16_qk_reduction=False,
    )
    output = wrapper.run(q.cuda(), (k.cuda(), v.cuda()), k_scale=k_scale, v_scale=v_scale)
    # output: [batch * seq_len, qo_head, headdim]. Convert to [batch, qo_head, seq_len, headdim]
    return output.view(batch, seq_len, qo_head, headdim).transpose(1, 2)
