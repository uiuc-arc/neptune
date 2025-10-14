from typing import cast

from tvm import te, tir
from tvm.tir.schedule import BlockRV, LoopRV
from tvm.topi.utils import get_const_tuple


def tile_innermost_n(sch: tir.Schedule, block, decisions: list[int]):
    loops = sch.get_loops(block)
    innermost_n = len(decisions)
    batches, loops = loops[:-innermost_n], loops[-innermost_n:]
    new_outer = tile_loops(sch, loops, decisions)
    return batches, new_outer


def tile_loops(
    sch: tir.Schedule, loops: list[LoopRV], decisions: list[int], inner_part_factor: int = 16
) -> list[LoopRV]:
    assert len(loops) == len(decisions)
    new_outer, new_inner = [], []
    for loop_var, factor in zip(loops, decisions):
        extent = get_loop_extent(sch, loop_var)
        if extent < factor:
            extents = [1, extent]
        else:
            assert extent % factor == 0, f"{extent} % {factor} != 0"
            outer, inner = extent // factor, factor
            extents = sch.sample_partitioned_tile(
                loop_var, 2, 1, inner_part_factor, decision=[outer, inner]
            )
        i0, i1 = sch.split(loop_var, extents)
        new_outer.append(i0)
        new_inner.append(i1)
    sch.reorder(*new_outer, *new_inner)
    return new_outer


def bind_block_idx(sch: tir.Schedule, loops: list[LoopRV]):
    block_indices = ["blockIdx.z", "blockIdx.y", "blockIdx.x"]
    assert len(loops) > 0
    while loops and len(block_indices) > 1:
        sch.bind(loops.pop(), block_indices.pop())
    # Here either `loops` is empty, or only 1 block label left.
    if len(loops) > 1:
        sch.bind(sch.fuse(*loops), block_indices[0])
    elif len(loops) == 1:
        sch.bind(loops[0], block_indices[0])


def annotate_triton_params(
    sch: tir.Schedule,
    block: BlockRV,
    n_warps_decision: int | None = None,
    n_stages_decision: int | None = None,
    n_warps_cands: list[int] = [1, 2, 4, 8],
    n_stages_cands: list[int] = [1, 2, 3, 4, 5, 6, 7],
):
    # NOTE: sample_categorical doesn't like being given a single choice.
    # The error doesn't show up until much later when we load a trace from JSON.
    # The probabilities is printed to JSON as `[1]`, and loaded back as an array of integers,
    # not floats, which causes the error.

    def find_index(decision: int | None, choices: list[int]):
        if decision is None:
            return None
        if decision not in choices:
            raise ValueError(f"Decision value {decision} not found in choices {choices}")
        return choices.index(decision)

    def equal_prob(choices: list[int]):
        return [1 / len(choices)] * len(choices)

    def sample(x: int | None, xs: list[int]):
        assert len(xs) >= 1
        if len(xs) == 1:
            assert x is None or x == xs[0]
            return xs[0]
        return sch.sample_categorical(xs, equal_prob(xs), find_index(x, xs))

    loop0 = sch.get_loops(block)[0]
    sch.annotate(loop0, "pragma_triton_num_warps", sample(n_warps_decision, n_warps_cands))
    sch.annotate(loop0, "pragma_triton_num_stages", sample(n_stages_decision, n_stages_cands))


def get_loop_extent(sch: tir.Schedule, loop: LoopRV) -> int:
    return int(cast(tir.For, sch.get(loop)).extent)


def batch_matmul(
    lhs, rhs, lhs_trans: bool = False, rhs_trans: bool = False, out_dtype: str = "float32"
):
    """A batched matmul that supports arbitrary batch dimensions."""
    if lhs_trans:
        *abatch, ak, ai = get_const_tuple(lhs.shape)
    else:
        *abatch, ai, ak = get_const_tuple(lhs.shape)
    if rhs_trans:
        *bbatch, bj, bk = get_const_tuple(rhs.shape)
    else:
        *bbatch, bk, bj = get_const_tuple(rhs.shape)
    assert ak == bk and abatch == bbatch, f"{lhs.shape} {rhs.shape}"
    k = te.reduce_axis((0, ak), name="k")

    def compute_impl(*ax):
        *ax, i, j = ax
        lhs_indices = (*ax, k, i) if lhs_trans else (*ax, i, k)
        rhs_indices = (*ax, j, k) if rhs_trans else (*ax, k, j)
        return te.sum(
            lhs[lhs_indices].astype(out_dtype) * rhs[rhs_indices].astype(out_dtype), axis=k
        )

    varargs_names = [f"b{i}" for i in range(len(abatch))] + ["i", "j"]
    ret = te.compute(
        abatch + [ai, bj],  # type: ignore
        compute_impl,
        name="T_batch_matmul_NT" if rhs_trans else "T_batch_matmul_NN",
        tag="batch_matmul",
        varargs_names=varargs_names,
    )
    return cast(te.Tensor, ret)
