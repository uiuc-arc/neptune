from tvm import te, tir
from tvm.ir.module import IRModule

from ..operators import BenchmarkOperator
from .attn_plain import (
    MaskModT,
    ScoreModT,
    general_attention,
    schedule_mask_attn_flash,
)
from .runner import NeptuneRunner


def create_grouped_query_attention(
    input_shape: tuple[int, int, int, int, int, int],
    score_mod: ScoreModT | None,
    mask_cond: MaskModT | None,
    func_name: str = "neptune_gqa",
) -> IRModule:
    """Compute grouped query attention (GQA) with customizable scoring and masking.

    This function implements grouped query attention where the number of query heads is a multiple
    of the number of key/value heads. Multiple query heads attend to a single key/value head.

    Parameters
    ----------
    input_shape : tuple[int, int, int, int, int, int]
        A 6-tuple (batch, q_heads, kv_heads, q_seq_len, kv_seq_len, head_dim).
    score_mod : see `general_attention`
    mask_cond : see `general_attention`

    Returns
    -------
    IRModule
        An IRModule with a single function that computes grouped query attention.
    """
    # NOTE: number of groups is the same as the number of key/value heads.
    B, QN, KVN, QS, KVS, D = input_shape
    assert QN % KVN == 0
    HPG = QN // KVN  # head per group
    q = te.placeholder((B, QN, QS, D), "float16", name="q")
    k = te.placeholder((B, KVN, KVS, D), "float16", name="k")
    v = te.placeholder((B, KVN, KVS, D), "float16", name="v")
    # This step transposes `q` on the HPG and KVN dimensions.
    # See here for reference:
    # https://github.com/fkodom/grouped-query-attention-pytorch/blob/main/grouped_query_attention_pytorch/attention.py#L88
    q_reshape = te.compute(
        (B, HPG, KVN, QS, D),
        lambda b, hpg, kvn, s, d: q(b, kvn * HPG + hpg, s, d),
        name="T_q_reshape",
    )
    k_broadcast = te.compute(
        (B, HPG, KVN, KVS, D), lambda b, g, h, ll, d: k(b, h, ll, d), name="T_k_broadcast"
    )
    v_broadcast = te.compute(
        (B, HPG, KVN, KVS, D), lambda b, g, h, ll, d: v(b, h, ll, d), name="T_v_broadcast"
    )
    output = general_attention(q_reshape, k_broadcast, v_broadcast, score_mod, mask_cond)
    output = te.compute(
        (B, QN, QS, D),
        lambda b, hg, ll, d: output(b, hg % HPG, hg // HPG, ll, d),
        name="T_output_reshape",
    )
    func = te.create_prim_func([q, k, v] + [output])
    return IRModule({func_name: func})


def create_gqa_decoding(
    input_shape: tuple[int, int, int, int, int],
    score_mod: ScoreModT | None,
    mask_cond: MaskModT | None,
    func_name: str = "neptune_gqa_decoding",
) -> IRModule:
    """A specialized version of `create_grouped_query_attention` for decoding.
    It uses repeated-loading tricks to pad Q to 16 heads, and arrange Q dimension in a way
    that K and V don't need to be reshaped."""

    B, QN, KVN, KVS, D = input_shape
    assert QN % KVN == 0
    HPG = QN // KVN  # head per group
    q = te.placeholder((B, QN, 1, D), "float16", name="q")
    k = te.placeholder((B, KVN, KVS, D), "float16", name="k")
    v = te.placeholder((B, KVN, KVS, D), "float16", name="v")
    q_reshape = te.compute(
        (B, KVN, HPG, D), lambda b, kvn, qn, d: q(b, kvn * HPG + qn, 0, d), name="T_q_reshape"
    )
    output = general_attention(q_reshape, k, v, score_mod, mask_cond)
    output = te.compute(
        (B, QN, 1, D),
        lambda b, hg, s, d: output(b, hg // HPG, hg % HPG, d),
        name="T_output_reshape",
    )
    func = te.create_prim_func([q, k, v] + [output])
    return IRModule({func_name: func})


def schedule_gqa(sch: tir.Schedule):
    sch.compute_inline("T_q_reshape")
    sch.compute_inline("T_k_broadcast")
    sch.compute_inline("T_v_broadcast")
    sch.reverse_compute_inline("T_output_reshape")
    return schedule_mask_attn_flash(sch)


def schedule_gqa_decoding(sch: tir.Schedule):
    from tvm.neptune.sch_utils import annotate_triton_params, bind_block_idx, get_loop_extent

    sch.compute_inline("T_q_reshape")
    sch.reverse_compute_inline("T_output_reshape")

    b0 = sch.get_block("T_batch_matmul_NT")
    b1 = sch.get_block("T_score_mod")
    b2 = sch.get_block("T_softmax_maxelem")
    b3 = sch.get_block("T_softmax_exp")
    b4 = sch.get_block("T_softmax_exp_cast")
    b5 = sch.get_block("T_batch_matmul_NN")
    b6 = sch.get_block("T_softmax_expsum")
    b7 = sch.get_block("T_softmax_norm")
    b8 = sch.get_block("T_cast")

    # Tile the first matmul, then apply split-k update.
    # Tile `j` with inner factor being min(128, extent(j)).
    *axes, i, j, k = sch.get_loops(b0)
    j_tile = min(128, get_loop_extent(sch, j))
    j0, j1 = sch.split(j, [None, j_tile])
    sch.reorder(j0, i, j1, k)
    bind_block_idx(sch, [*axes, j0])
    sch.reverse_compute_at(b1, j0, preserve_unit_loops=True)
    b2rf = sch.split_k_update(b2, j0, factor_axis=0)
    sch.split_k_update(b6, j0, factor_axis=0)
    sch.split_k_update(b5, j0, factor_axis=0)
    # Cache write `T_softmax_maxelem_rf`, because b3 still reads from it.
    sch.reverse_compute_at(sch.cache_write(b2rf, 0, "shared", consumer_blocks=[b3]), j0)

    # Fuse the "global" loop nests into one kernel.
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
    annotate_triton_params(sch, b0, 4, 2, n_stages_cands=[1, 2, 3, 4])
    annotate_triton_params(sch, b2, 4, 1, n_stages_cands=[1, 2, 3, 4])
    sch.compact_buffer()
    sch.to_tile_expr_form([])

    return sch


class NeptuneGQARunner(NeptuneRunner):
    @classmethod
    def create_flex_from_schedulers(  # type: ignore
        cls,
        operator: BenchmarkOperator,
        score_mod: ScoreModT | None = None,
        mask_cond: MaskModT | None = None,
        schedulers: tuple = (schedule_gqa,),
    ):
        return super()._create_from_schedulers(
            operator,
            lambda shape: create_grouped_query_attention(shape, score_mod, mask_cond),
            schedulers,
        )

    def extract_shape(self, inputs) -> tuple:
        q, k, v = inputs
        B, QN, QS, D = q.shape
        _, KVN, KVS, _ = k.shape
        return (B, QN, KVN, QS, KVS, D)


class NeptuneGQADecodeRunner(NeptuneRunner):
    @classmethod
    def create_flex_from_schedulers(  # type: ignore
        cls,
        operator: BenchmarkOperator,
        score_mod: ScoreModT | None = None,
        schedulers: tuple = (schedule_gqa_decoding,),
    ):
        return super()._create_from_schedulers(
            operator,
            lambda shape: create_gqa_decoding(shape, score_mod, None),
            schedulers,
        )

    def extract_shape(self, inputs) -> tuple:
        q, k, v = inputs
        B, QN, QS, D = q.shape
        assert QS == 1, "Decoding requires Q seq_len 1"
        _, KVN, KVS, _ = k.shape
        return (B, QN, KVN, KVS, D)
