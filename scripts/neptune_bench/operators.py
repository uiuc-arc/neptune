from abc import abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BenchmarkOperator:
    name: str

    @abstractmethod
    def create_inputs(self, shape: tuple) -> tuple:
        pass

    @abstractmethod
    def complete_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        pass

    @abstractmethod
    def flops(self, shape: tuple) -> float:
        pass

    def with_name(self, name: str) -> "BenchmarkOperator":
        sdict = self.__dict__.copy()
        sdict["name"] = name
        return self.__class__(**sdict)


@dataclass(frozen=True)
class QKVInputOp(BenchmarkOperator):
    q_heads: int
    head_dim: int
    n_groups: int | None = None
    is_causal: bool = True
    is_decode: bool = False
    q_dtype: torch.dtype = torch.float16
    kv_dtype: torch.dtype = torch.float16

    def complete_shape(self, input_shape: tuple) -> tuple:
        assert len(input_shape) == 2, (
            "QKV input shape must be (batch, seq_len). "
            "q_heads, head_dim, n_groups are baked in to the operator definition."
        )
        batch, seq_len = input_shape
        q_seqlen = 1 if self.is_decode else seq_len
        kv_heads = self.n_groups or self.q_heads
        return (batch, self.q_heads, kv_heads, q_seqlen, seq_len, self.head_dim)

    def create_inputs(self, shape: tuple) -> tuple:
        batch, q_heads, kv_heads, q_seqlen, seq_len, head_dim = shape
        q_shape = (batch, q_heads, q_seqlen, head_dim)
        kv_shape = (batch, kv_heads, seq_len, head_dim)
        return (
            torch.randn(q_shape, dtype=self.q_dtype, device="cuda"),
            torch.randn(kv_shape, dtype=self.kv_dtype, device="cuda"),
            torch.randn(kv_shape, dtype=self.kv_dtype, device="cuda"),
        )

    def flops(self, shape: tuple) -> float:
        b, qh, _, qsl, kvsl, h = shape
        # 2BNS^2H per matmul, 4BNS^2 for the softmax
        flops = 4 * b * qh * (qsl * kvsl) * (h + 1)
        if self.is_causal and not self.is_decode:
            flops /= 2
        return flops


@dataclass(frozen=True)
class SoftCapAttnOp(QKVInputOp):
    softcap: float = 50.0


@dataclass(frozen=True)
class WindowedAttnOp(QKVInputOp):
    window_size: int = 4096

    def flops(self, shape: tuple) -> float:
        b, qh, _, qsl, kvsl, h = shape
        assert qsl == kvsl, "Windowed attention only supports square inputs"
        seq_len_square = qsl * min(qsl, self.window_size)
        return 2 * b * qh * seq_len_square * (h + 1)


class AlibiOp(QKVInputOp):
    def create_inputs(self, shape: tuple) -> tuple:
        q, k, v = super().create_inputs(shape)
        return (q, k, v, torch.randn((self.q_heads,), device="cuda"))


# Models ViT-L. Plain global attention.
#   Impl: `transformers.models.vit.modeling_vit.ViTSelfAttention`
#   References:
#     https://arxiv.org/pdf/2010.11929, Table 1.
# NOTE: in ViT, the number of tokens is based on the height / width of the image,
# so it's somewhat constrained. Now we still allow the user to feed in the number of tokens.
# seq_len = (H / 16) * (W / 16) + 1.
PF_GLOBAL = QKVInputOp("prefill_global", q_heads=32, head_dim=128, is_causal=False)
# Models GPT3 6.7B (curie).
#   Features: mask (causal)
#   References:
#     https://arxiv.org/pdf/2005.14165, Table 2.1;
#     https://en.wikipedia.org/wiki/GPT-3#GPT-3_models (same table)
PF_CAUSAL = QKVInputOp("prefill_causal", q_heads=32, head_dim=128)
# Models MPT 7B.
#   Features: score (alibi); mask (causal)
#   Impl: `transformers.models.mpt.modeling_mpt.MPTAttention`
#   References:
#     https://huggingface.co/mosaicml/mpt-7b
#     https://arxiv.org/pdf/2108.12409 (the original Alibi paper, not MPT)
PF_ALIBI = AlibiOp("prefill_alibi", q_heads=32, head_dim=128)
# Models LLama3 70B.
#   Features: heads (GQA); mask (causal)
#   Impl: `transformers.models.llama.modeling_llama.LlamaAttention`
#   References:
#     https://arxiv.org/pdf/2302.13971
PF_GQA = QKVInputOp("prefill_gqa", q_heads=64, head_dim=128, n_groups=8)
# Models Gemma2 27B.
#   Features: score (softcap); heads (GQA); mask (causal)
#   Impl: `transformers.models.gemma2.modeling_gemma2.Gemma2Attention`
#   References:
#     https://huggingface.co/blog/gemma2
#     https://arxiv.org/pdf/2408.00118
# NOTE: Gemma2 by default only supports context length upto 8192 tokens.
# TODO: this used to be Gemma2 9B, which has head_dim=256 and takes too much SMEM on some GPUs.
#   we should revisit the 9B case though and compare with PyTorch MemEfficient kernel which supports it.
PF_SOFTCAP = SoftCapAttnOp("prefill_softcap", q_heads=32, head_dim=128, n_groups=16, softcap=50.0)
# Models Gemma2 27B (windowed layers). Every other layer in Gemma2 is windowed with size 4096.
#   Features: mask (windowed)
PF_WINDOWED = WindowedAttnOp(
    "prefill_windowed", q_heads=32, head_dim=128, n_groups=8, window_size=4096
)
# TODO: add RoPE. RoPE can be seen as a separate operation before the attention proper --
# that's how we modelled Llama3 without RoPE, but it would be good to have a variant where it's fused into attention.

# Attention in decoding. Distinctions on the mask during prefill no longer apply to decoding,
# so out of windowed, global and causal, we only select causal.
DC_CAUSAL = QKVInputOp("decode_causal", q_heads=32, head_dim=128, is_decode=True)
DC_ALIBI = AlibiOp("decode_alibi", q_heads=32, head_dim=128, is_decode=True)
DC_GQA = QKVInputOp("decode_gqa", q_heads=64, head_dim=128, n_groups=8, is_decode=True)
DC_SOFTCAP = SoftCapAttnOp(
    "decode_softcap", q_heads=32, head_dim=128, n_groups=16, softcap=50.0, is_decode=True
)
