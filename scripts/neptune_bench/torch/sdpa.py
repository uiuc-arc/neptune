from enum import Enum
from functools import partial

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

from ..runner import BenchmarkOperator, DirectOutputRunner
from ..utils import bnsh_expand_for_gqa, get_current_gpu_info


class AttentionAlgo(Enum):
    UNFUSED = "unfused"  # Called "math" kernel in PyTorch
    UNFUSED_COMPILED = "unfused-compiled"
    MEM_EFFICIENT = "mem-eff"
    CUDNN = "cudnn"
    FLASH = "flash"

    def __str__(self):
        return self.value

    def get_backend_and_compile_flag(self):
        import os

        match self:
            case AttentionAlgo.CUDNN:
                os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
                return SDPBackend.CUDNN_ATTENTION, False
            case AttentionAlgo.FLASH:
                return SDPBackend.FLASH_ATTENTION, False
            case AttentionAlgo.MEM_EFFICIENT:
                return SDPBackend.EFFICIENT_ATTENTION, False
            case AttentionAlgo.UNFUSED:
                return SDPBackend.MATH, False
            case AttentionAlgo.UNFUSED_COMPILED:
                return SDPBackend.MATH, True

    def get_language(self):
        match self:
            case AttentionAlgo.UNFUSED:
                return None
            case AttentionAlgo.UNFUSED_COMPILED:
                return "triton"
            case AttentionAlgo.MEM_EFFICIENT:
                return "cutlass"
            case AttentionAlgo.CUDNN:
                return "cudnn"
            case AttentionAlgo.FLASH:
                return "cutlass"


@torch.compile(backend="inductor", options={"triton.cudagraphs": True})
def run_attention_compiled(q, k, v, is_causal: bool, is_gqa: bool):
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal, enable_gqa=is_gqa
    )


class TorchSDPARunner(DirectOutputRunner):
    def __init__(
        self,
        operator: BenchmarkOperator,
        algo: AttentionAlgo,
        is_causal: bool,
        is_gqa: bool = False,
    ):
        super().__init__(operator, "torch", language=algo.get_language(), impl_name=algo.value)
        self.backend, compile = algo.get_backend_and_compile_flag()
        self.runner = (
            partial(run_attention_compiled, is_causal=is_causal, is_gqa=is_gqa)
            if compile
            else partial(
                F.scaled_dot_product_attention,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
                enable_gqa=is_gqa,
            )
        )

    def setup(self, *inputs) -> bool:
        if get_current_gpu_info().is_amd and self.backend == SDPBackend.CUDNN_ATTENTION:
            return False
        self.q, self.k, self.v = inputs
        return True

    def run_output(self) -> torch.Tensor:
        with torch.nn.attention.sdpa_kernel(backends=[self.backend]), self.mark_ctx():
            return self.runner(self.q, self.k, self.v)


@torch.compile(backend="inductor", options={"triton.cudagraphs": True})
def run_decode_gqa_compiled(q, k, v):
    q, k, v = bnsh_expand_for_gqa(q, k, v)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)


# Only one known setup supports GQA + decoding: simulate GQA by repeating the key and value tensors,
# and apply mem-efficient backend. We can also use some compilation to try to make it faster.
class TorchDecodeGQARunner(DirectOutputRunner):
    def __init__(self, operator: BenchmarkOperator):
        super().__init__(
            operator, "torch", language="cutlass", impl_name=AttentionAlgo.MEM_EFFICIENT.value
        )

    def setup(self, *inputs) -> bool:
        self.q, self.k, self.v = inputs
        return True

    def run_output(self) -> torch.Tensor:
        with (
            torch.nn.attention.sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]),
            self.mark_ctx(),
        ):
            return run_decode_gqa_compiled(self.q, self.k, self.v)
