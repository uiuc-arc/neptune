from logging import getLogger

import torch

from ..runner import BenchmarkOperator, BenchmarkRunner

logger = getLogger(__name__)


class FlashInferRunner(BenchmarkRunner):
    try:
        import flashinfer as fi
    except ImportError:
        fi = None

    def __init__(self, operator: BenchmarkOperator, is_causal: bool, is_decode: bool):
        super().__init__(operator, "flashinfer", language="cutlass")
        self.is_causal = is_causal
        self.is_decode = is_decode
        assert not (is_causal and is_decode), "FlashInfer decode cannot be causal"
        self.fi_wrapper = None

    def setup(self, *inputs) -> bool:
        if self.fi is None:
            logger.warning("flashinfer is not installed")
            return False

        q, self.k, self.v = inputs
        self.q_shape = batch, q_heads, q_sl, head_dim = q.shape
        _, kv_heads, k_sl, _ = self.k.shape
        if self.is_decode:
            assert q_sl == 1, "Decoding requires Q seq_len 1"
        assert self.k.shape == self.v.shape == (batch, kv_heads, k_sl, head_dim)
        # Prepare data for FlashInfer's paged API
        # Q: [batch_size * seq_len, num_heads, head_dim]
        self.q = q.transpose(1, 2).reshape(-1, q_heads, head_dim).contiguous()

        # Set up paged KV cache parameters. We're not actually using a paged KV cache,
        # so we just set up the parameters so that everything fits in one page.
        page_size = k_sl
        kv_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=q.device)
        kv_indices = torch.arange(0, batch, dtype=torch.int32, device=q.device)
        kv_last_page_len = torch.full((batch,), k_sl, dtype=torch.int32, device=q.device)
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=q.device)
        if self.is_decode:
            self.fi_wrapper = self.fi.decode.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer, kv_layout="HND"
            )
            self.fi_wrapper.plan(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                q_heads,
                kv_heads,
                head_dim,
                page_size,
                q_data_type=self.q.dtype,
                kv_data_type=self.k.dtype,
            )
        else:
            self.fi_wrapper = self.fi.prefill.BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer, kv_layout="HND"
            )
            q_indptr = torch.arange(
                0, (batch + 1) * k_sl, step=k_sl, dtype=torch.int32, device=q.device
            )
            self.fi_wrapper.plan(
                q_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                q_heads,
                kv_heads,
                head_dim,
                page_size,
                causal=self.is_causal,
                q_data_type=self.q.dtype,
                kv_data_type=self.k.dtype,
            )
        return True

    def run(self):
        assert self.fi_wrapper is not None
        with self.mark_ctx():
            self.output = self.fi_wrapper.run(self.q, (self.k, self.v))

    def get_output(self) -> torch.Tensor:
        assert self.output is not None, "Output not set"
        assert self.q_shape is not None
        # Reshape back to [batch_size, num_heads, seq_len, head_dim]
        batch_size, num_heads, seq_len, head_dim = self.q_shape
        return self.output.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


class FlashInferFP8PrefillRunner(BenchmarkRunner):
    def __init__(self, operator: BenchmarkOperator, is_causal: bool):
        super().__init__(operator, "flashinfer", language="cutlass")
        self.is_causal = is_causal

    def setup(self, *inputs) -> bool:
        q, k, v, self.k_scale, self.v_scale = inputs
        self.q, self.k, self.v = map(_fi_transform_input, (q, k, v))
        return True

    def run(self):
        from .paged import attention_fp8_paged

        with self.mark_ctx():
            self.output = attention_fp8_paged(
                self.q, self.k, self.v, self.k_scale, self.v_scale, is_causal=self.is_causal
            )

    def get_output(self) -> torch.Tensor:
        assert self.output is not None, "Output not set"
        return _fi_transform_output(self.output)  # type: ignore


def _fi_transform_input(t: torch.Tensor):
    assert t.shape[0] == 1, "flashinfer only supports batch size 1"
    return torch.permute(t[0], (1, 0, 2)).contiguous()


def _fi_transform_output(t: torch.Tensor):
    return torch.unsqueeze(torch.permute(t, (1, 0, 2)), 0)
