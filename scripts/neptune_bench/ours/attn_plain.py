from typing import Callable, Sequence

import numpy as np
from tvm import ir, te, tir
from tvm.neptune.sch_utils import (
    annotate_triton_params,
    batch_matmul,
    bind_block_idx,
    tile_innermost_n,
    tile_loops,
)

from .runner import BenchmarkOperator, NeptuneRunner
from .softmax import softmax

MaskModT = Callable[..., tir.PrimExpr | bool]
ScoreModT = Callable[..., tir.PrimExpr | float]


def causal_mask_cond(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def general_attention(
    q: te.Tensor,
    k: te.Tensor,
    v: te.Tensor,
    score_mod: ScoreModT | None,
    mask_cond: MaskModT | None,
    v_scale: te.Tensor | None = None,
) -> te.Tensor:
    """Apply a general attention kernel on the inputs Q, K, V, with customizable scoring and masking.
    This function has an API that's similar to FlexAttention.

    Parameters
    ----------
    q : te.Tensor
        Query tensor with shape [*batch, num_heads, q_seq_len, head_dim]
    k : te.Tensor
        Key tensor with shape [*batch, num_heads, kv_seq_len, head_dim]
        kv_seq_len can differ from q_seq_len (useful for decoding).
    v : te.Tensor
        Value tensor with shape [*batch, num_heads, kv_seq_len, head_dim]
    score_mod : ScoreModT
        A function that modifies attention scores. Takes parameters:
        (score, *indices) and returns modified score
    mask_cond : MaskModT
        A function that determines masking. Takes parameters:
        (*indices) and returns boolean mask
    v_scale : te.Tensor | None
        Scale factor for the value tensor. If provided, the output will be multiplied by this tensor.

    Returns
    -------
    te.Tensor
        Output tensor with shape [*batch, num_heads, seq_len, head_dim]
    """

    def score_mod_(score, *args):
        head_dim = int(q.shape[-1])
        score = score / np.sqrt(head_dim)
        if score_mod is not None:
            score = score_mod(score, *args)
        return score

    if mask_cond is None:
        mask_cond = lambda *_: True  # noqa: E731
    p = batch_matmul(q, k, rhs_trans=True, out_dtype="float32")
    s = te.compute(
        p.shape,
        lambda *ax: tir.if_then_else(
            mask_cond(*ax), score_mod_(p(*ax), *ax), tir.min_value("float32")
        ),
        name="T_score_mod",
    )
    # Compute the elem-wise exp and the sum with a helper function.
    s_exp, s_expsum = softmax(s)
    # Cast `s_exp` to a smaller dtype, then do the second matmul.
    s_exp = te.compute(
        p.shape, lambda *axes: s_exp(*axes).astype(q.dtype), name="T_softmax_exp_cast"
    )
    sv = batch_matmul(s_exp, v, rhs_trans=False, out_dtype="float32")
    # Commute `/ expsum` after the matmul.
    sv = te.compute(sv.shape, lambda *axes: sv(*axes) / s_expsum(*axes[:-1]), name="T_softmax_norm")
    v_scale = 1.0 if v_scale is None else v_scale
    return te.compute(sv.shape, lambda *i: (v_scale * sv(*i)).astype(q.dtype), name="T_cast")  # type: ignore


def create_general_attention(
    input_shape: tuple[int, int, int, int, int],
    score_mod: ScoreModT | None = None,
    mask_cond: MaskModT | None = None,
    # TODO: to allow score_mod and mask_cond to capture tensors, create an analysis like relax.analysis.free_vars
    # so we don't have to pass tensors to `create_prim_func` explicitly.
    extra_tensors: Sequence[te.Tensor] = (),
    func_name: str = "neptune_attention",
):
    """Create a general attention kernel from input shape. This function has an API that's similar to FlexAttention.
    This function covers both prefill and decode, but not GQA.

    Parameters
    ----------
    input_shape : tuple[int, int, int, int, int]
        A 5-tuple (batch, num_heads, q_seq_len, kv_seq_len, head_dim).
    score_mod : see `general_attention`
    mask_cond : see `general_attention`
    extra_tensors : Sequence[te.Tensor]
        Extra tensors to pass to the function. For example, alibi bias for alibi attention.
    func_name : str
        The name of the attention function in the output IRModule.

    Returns
    -------
    IRModule
        An IRModule with a single function that computes general attention.
    """
    b, n, qs, kvs, h = input_shape
    q = te.placeholder((b, n, qs, h), "float16", name="q")
    k = te.placeholder((b, n, kvs, h), "float16", name="k")
    v = te.placeholder((b, n, kvs, h), "float16", name="v")
    o = general_attention(q, k, v, score_mod, mask_cond)
    func = te.create_prim_func([q, k, v] + list(extra_tensors) + [o])
    return ir.IRModule({func_name: func})


def create_alibi_attention(input_shape: tuple[int, int, int, int, int]):
    nh = input_shape[1]
    alibi_bias = te.placeholder([nh], "float32", name="alibi_bias")
    return create_general_attention(
        input_shape,
        # score - alibi_bias[h] * abs(kv_idx - q_idx) -- to match FlashAttn official impl
        score_mod=lambda score, b, h, q_idx, kv_idx: score + alibi_bias[h] * (kv_idx - q_idx),
        mask_cond=causal_mask_cond,
        extra_tensors=(alibi_bias,),
        func_name="neptune_alibi_attention",
    )


def schedule_attention_plain(sch: tir.Schedule):
    b0 = sch.get_block("T_batch_matmul_NT")
    b1 = sch.get_block("T_score_mod")
    b2 = sch.get_block("T_softmax_maxelem")
    b3 = sch.get_block("T_softmax_exp")
    b4 = sch.get_block("T_softmax_exp_cast")
    b5 = sch.get_block("T_batch_matmul_NN")
    b6 = sch.get_block("T_softmax_expsum")
    b7 = sch.get_block("T_softmax_norm")
    b8 = sch.get_block("T_cast")

    # Tile the first matmul, and fuse score_mod under it.
    axes, (i0, j0, mm1_k0) = tile_innermost_n(sch, b0, [128, 64, 128])
    sch.reverse_compute_at(b1, j0)
    sch.set_scope(b0, 0, "shared")
    bind_block_idx(sch, [*axes, i0, j0])

    # Tile the max part of softmax, apply rfactor, then combine the two reduction blocks.
    *axes, i, j = sch.get_loops(b2)
    (i0,) = tile_loops(sch, [i], [2], inner_part_factor=1)
    bind_block_idx(sch, [*axes, i0])
    sch.reverse_compute_at(b3, i0)
    sch.reverse_compute_at(b4, i0)
    sch.reverse_compute_at(b6, i0)
    sch.set_scope(b2, 0, "shared")
    sch.set_scope(b3, 0, "shared")

    axes, (i0, j0, mm2_k0) = tile_innermost_n(sch, b5, [128, 64, 128])
    sch.set_scope(b5, 0, "shared")
    sch.reverse_compute_at(b7, j0)
    sch.set_scope(b7, 0, "shared")
    sch.reverse_compute_at(b8, j0)
    bind_block_idx(sch, [*axes, i0, j0])

    sch.decompose_reduction(b0, mm1_k0)
    sch.decompose_reduction(b5, mm2_k0)
    annotate_triton_params(sch, b0, 4, 2)
    annotate_triton_params(sch, b2, 4, 2)
    annotate_triton_params(sch, b5, 4, 2)
    sch.compact_buffer()
    # Request blockization between k0 and the inner loops for each matmul.
    sch.to_tile_expr_form([mm1_k0, mm2_k0])

    return sch


def _schedule_attention_flash(sch: tir.Schedule, default_blk_sizes: tuple[int, int]):
    b0 = sch.get_block("T_batch_matmul_NT")
    b1 = sch.get_block("T_score_mod")
    b2 = sch.get_block("T_softmax_maxelem")
    b3 = sch.get_block("T_softmax_exp")
    b4 = sch.get_block("T_softmax_exp_cast")
    b5 = sch.get_block("T_batch_matmul_NN")
    b6 = sch.get_block("T_softmax_expsum")
    b7 = sch.get_block("T_softmax_norm")
    b8 = sch.get_block("T_cast")

    # Don't tile `k` as that conflicts with cache-reading of `q`.
    *axes, i, j, k = sch.get_loops(b0)
    i0, j0 = tile_loops(sch, [i, j], list(default_blk_sizes))
    sch.compute_at(sch.cache_read(b0, 0, "shared"), i0)
    bind_block_idx(sch, [*axes, i0])
    sch.reverse_compute_at(b1, j0)

    b2rf = sch.rolling_update(b2, j0, factor_axis=0)
    # b6 shows up in the program after b5, but we want to compute it first.
    b6rf = sch.rolling_update(b6, j0, factor_axis=0)
    b5rf = sch.rolling_update(b5, j0, factor_axis=0)
    sch.reverse_compute_at(b7, i0)
    sch.reverse_compute_at(b8, i0)
    for blk in [b0, b1, b2, b2rf, b3, b4, b5, b5rf, b6, b6rf, b7]:
        sch.set_scope(blk, 0, "shared")

    sch.split_scan_buffer(b2, j0, 0)
    sch.decompose_reduction(b5, j0)
    sch.decompose_reduction(b6, j0)

    return b0, b1, j0


def schedule_full_attn_flash(sch: tir.Schedule):
    b0, _, _ = _schedule_attention_flash(sch, (128, 32))
    annotate_triton_params(sch, b0, 4, 2)
    sch.compact_buffer()
    sch.to_tile_expr_form([])
    return sch


def schedule_mask_attn_flash(sch: tir.Schedule):
    b0, b1, j0 = _schedule_attention_flash(sch, (128, 64))
    sch.propagate_if_then_else(b1, j0, "")
    annotate_triton_params(sch, b0, 8, 3)
    sch.compact_buffer()
    sch.to_tile_expr_form([])
    return sch


def schedule_flash_decoding(sch: tir.Schedule):
    b0 = sch.get_block("T_batch_matmul_NT")
    b1 = sch.get_block("T_score_mod")
    b2 = sch.get_block("T_softmax_maxelem")
    b3 = sch.get_block("T_softmax_exp")
    b4 = sch.get_block("T_softmax_exp_cast")
    b5 = sch.get_block("T_batch_matmul_NN")
    b6 = sch.get_block("T_softmax_expsum")
    b7 = sch.get_block("T_softmax_norm")
    b8 = sch.get_block("T_cast")

    *axes, j, k = sch.get_loops(b0)
    (j0, b0k0) = tile_loops(sch, [j, k], [128, 128])
    bind_block_idx(sch, [*axes, j0])
    sch.reverse_compute_at(b1, j0, preserve_unit_loops=True)
    b2rf = sch.split_k_update(b2, j0, factor_axis=0)
    sch.split_k_update(b6, j0, factor_axis=0)
    sch.split_k_update(b5, j0, factor_axis=0)
    sch.reverse_compute_at(sch.cache_write(b2rf, 0, "shared", consumer_blocks=[b3]), j0)

    # i0 is the loop over the rows, which has size 1. We put it outside so that it's easier to
    # fuse it with the `axes` loops later.
    j0, *axes, i0 = sch.get_loops(b2)
    assert len(axes) > 0
    sch.reorder(i0, *axes, j0)
    sch.reverse_compute_at(b6, axes[-1])
    sch.reverse_compute_at(b5, axes[-1])
    sch.reverse_compute_at(b7, axes[-1])
    sch.reverse_compute_at(b8, axes[-1])
    sch.bind(sch.fuse(i0, *axes), "blockIdx.x")

    for blk in [b0, b1, b2, b2rf, b3, b4, b5, b6, b7]:
        sch.set_scope(blk, 0, "shared")
    sch.decompose_reduction(b0, b0k0)
    annotate_triton_params(sch, b0, 4, 2, n_stages_cands=[1, 2, 3, 4])
    annotate_triton_params(sch, b2, 4, 1, n_stages_cands=[1, 2, 3, 4])
    sch.compact_buffer()
    sch.to_tile_expr_form([b0k0])
    return sch


class NeptuneAttentionRunner(NeptuneRunner):
    @classmethod
    def create_flex_from_schedulers(  # type: ignore
        cls,
        operator: BenchmarkOperator,
        score_mod: ScoreModT | None = None,
        mask_cond: MaskModT | None = None,
        schedulers: tuple | None = None,
    ):
        """Create multiple runners with different schedulers.
        Restricts to create_general_attention, but allows customizing the score_mod and mask_cond."""
        return cls.create_from_schedulers(
            operator,
            lambda shape: create_general_attention(shape, score_mod, mask_cond),
            mask_cond is not None,
            schedulers,
        )

    @classmethod
    def create_from_schedulers(  # type: ignore
        cls,
        operator: BenchmarkOperator,
        mod_creator: Callable[[tuple], ir.IRModule],
        has_mask: bool,
        schedulers: tuple | None = None,
    ):
        """Create multiple runners with different schedulers. Define any computation in `mod_creator`."""
        if schedulers is None:
            flash_sch = schedule_full_attn_flash if not has_mask else schedule_mask_attn_flash
            schedulers = (schedule_attention_plain, flash_sch)
        return super()._create_from_schedulers(operator, mod_creator, schedulers)

    def extract_shape(self, inputs) -> tuple:
        q, k, v, *_ = inputs
        assert (kvs := k.shape[2]) == v.shape[2]
        b, n, qs, h = q.shape
        return (b, n, qs, kvs, h)

    def _size_unsupported(self, shape: tuple) -> bool:
        b, n, qs, kvs, h = shape
        return self.scheduler == schedule_attention_plain and (qs >= 1024 and kvs >= 1024)


# We have a separate decode runner because we want to check Q's seq_len is actually 1,
# and we can skip any Q/K masking.
class NeptuneDecodeRunner(NeptuneRunner):
    @classmethod
    def create_flex_from_schedulers(  # type: ignore
        cls,
        operator: BenchmarkOperator,
        score_mod: ScoreModT | None = None,
        schedulers: tuple = (schedule_flash_decoding,),
    ):
        return cls.create_from_schedulers(
            operator, lambda shape: create_general_attention(shape, score_mod), schedulers
        )

    @classmethod
    def create_from_schedulers(  # type: ignore
        cls,
        operator: BenchmarkOperator,
        mod_creator: Callable[[tuple], ir.IRModule],
        schedulers: tuple = (schedule_flash_decoding,),
    ):
        return super()._create_from_schedulers(operator, mod_creator, schedulers)

    def extract_shape(self, inputs) -> tuple:
        q, k, v, *_ = inputs
        b, n, qs, h = q.shape
        assert qs == 1, "Decoding requires Q seq_len 1"
        _, _, kvs, _ = k.shape
        return (b, n, 1, kvs, h)
