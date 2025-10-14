from typing import Callable

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from ..runner import BenchmarkOperator, DirectOutputRunner

flex_attn_compiled: Callable[..., torch.Tensor] = torch.compile(flex_attention)  # type: ignore


class FlexAttentionRunner(DirectOutputRunner):
    def __init__(
        self,
        operator: BenchmarkOperator,
        mask_expr: Callable | None = None,
        score_mod: Callable | None = None,
        enable_gqa: bool = False,
    ):
        super().__init__(operator, "flex", language="triton")
        self.mask_expr = mask_expr
        self.score_mod = score_mod
        self.enable_gqa = enable_gqa
        self.block_mask: BlockMask | None = None

    def setup(self, *inputs) -> bool:
        if self.mask_expr is not None:
            seq_len = inputs[0].shape[2]
            self.block_mask = create_block_mask(
                self.mask_expr, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len
            )
        self.q, self.k, self.v = inputs
        return True

    def run_output(self) -> torch.Tensor:
        with self.mark_ctx():
            return flex_attn_compiled(
                self.q,
                self.k,
                self.v,
                block_mask=self.block_mask,
                score_mod=self.score_mod,
                enable_gqa=self.enable_gqa,
            )


class FlexAlibiRunner(FlexAttentionRunner):
    def __init__(self, operator: BenchmarkOperator, is_decode: bool):
        super().__init__(
            operator,
            mask_expr=None if is_decode else lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
        )

    def setup(self, *inputs):
        self.q, self.k, self.v, self.alibi_bias = inputs
        return True

    def run_output(self) -> torch.Tensor:
        def score_mod(score, b, h, q_idx, kv_idx):
            return score + self.alibi_bias[h] * (kv_idx - q_idx)  # type: ignore

        with self.mark_ctx():
            return flex_attn_compiled(
                self.q, self.k, self.v, score_mod=score_mod, block_mask=self.block_mask
            )
