import torch

from ..runner import BenchmarkOperator, BenchmarkRunner, DirectOutputRunner
from ..utils import bnsh_expand_for_gqa, get_current_gpu_info
from .openai import attention as openai_attn
from .tri_dao import fused_attn as tri_dao_general_attn


class OpenAITritonRunner(DirectOutputRunner):
    def __init__(self, operator: BenchmarkOperator, is_causal: bool):
        super().__init__(operator, "triton", language="triton", impl_name="openai")
        self.is_causal = is_causal

    def setup(self, *inputs) -> bool:
        self.q, self.k, self.v = inputs
        self.gqa = self.k.shape[1] != self.q.shape[1]
        return True

    def run_output(self) -> torch.Tensor:
        with self.mark_ctx():
            if self.gqa:
                q, k, v = bnsh_expand_for_gqa(self.q, self.k, self.v)
            else:
                q, k, v = self.q, self.k, self.v
            return openai_attn(q, k, v, causal=self.is_causal)


class TriDaoTritonRunner(BenchmarkRunner):
    def __init__(self, operator: BenchmarkOperator, is_causal: bool):
        super().__init__(operator, "triton", language="triton", impl_name="tri-dao")
        self.is_causal = is_causal
        self.is_amd = get_current_gpu_info().is_amd
        self.output = None

    def setup(self, *inputs) -> bool:
        q, k, v = inputs
        self.gqa = k.shape[1] != q.shape[1]
        self.q, self.k, self.v = map(_transform_io, (q, k, v))
        return True

    def run(self):
        if self.is_amd:
            import flash_attn.flash_attn_triton_amd.interface_fa as amd_fa
            from flash_attn.flash_attn_triton_amd.utils import MetaData

            _, seq_len, _, head_dim = self.q.shape
            meta = MetaData(sm_scale=head_dim**-0.5)
            meta.max_seqlens_q = seq_len
            meta.max_seqlens_k = seq_len
            meta.layout = "bshd"
            # Allocate output
            self.output = torch.empty_like(self.q)
            with self.mark_ctx():
                # This set of parameters are for version 2.8.1.
                # We use 2.7.4 on NVIDIA GPUs and 2.8.1 on AMD GPUs.
                amd_fa.fwd(  # type: ignore
                    self.q,
                    self.k,
                    self.v,
                    self.output,
                    alibi_slopes=None,
                    dropout_p=0.0,
                    softmax_scale=head_dim**-0.5,
                    causal=self.is_causal,
                    window_size_left=0,  # Window size doesn't do anything yet.
                    window_size_right=0,
                    softcap=0.0,  # Neither does softcap.
                    return_softmax=False,
                )
        else:
            with self.mark_ctx():
                if self.gqa:
                    q, k, v = bsnh_expand_for_gqa(self.q, self.k, self.v)
                else:
                    q, k, v = self.q, self.k, self.v
                self.output = tri_dao_general_attn(q, k, v, causal=self.is_causal)

    def get_output(self) -> torch.Tensor:
        assert self.output is not None, "Output not set"
        return _transform_io(self.output)


class XformerDecodeRunner(BenchmarkRunner):
    def __init__(self, operator: BenchmarkOperator):
        super().__init__(operator, "triton", language="triton", impl_name="xformer")

    def setup(self, *inputs) -> bool:
        q, k, v = inputs
        self.gqa = k.shape[1] != q.shape[1]
        self.q, self.k, self.v = map(_transform_io, (q, k, v))
        return True

    def run(self):
        import xformers.ops as xops

        # Force the Triton Split-K implementation
        with self.mark_ctx():
            if self.gqa:
                q, k, v = bsnh_expand_for_gqa(self.q, self.k, self.v)
            else:
                q, k, v = self.q, self.k, self.v
            self.output = xops.memory_efficient_attention(
                q, k, v, op=[xops.fmha.triton_splitk.FwOp]
            )

    def get_output(self) -> torch.Tensor:
        assert self.output is not None, "Output not set"
        return _transform_io(self.output)


def _transform_io(t: torch.Tensor):
    return torch.permute(t, (0, 2, 1, 3))


def bsnh_expand_for_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    q_heads = q.shape[2]
    b, s, kv_heads, d = k.shape
    groups = q_heads // kv_heads
    kv_shape = (b, s, q_heads, d)
    k = k.unsqueeze(3).expand(-1, -1, -1, groups, -1).contiguous().view(kv_shape)
    v = v.unsqueeze(3).expand(-1, -1, -1, groups, -1).contiguous().view(kv_shape)
    return q, k, v
