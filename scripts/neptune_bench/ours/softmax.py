from tvm import te, tir
from tvm.ir.module import IRModule
from tvm.neptune.sch_utils import (
    annotate_triton_params,
    bind_block_idx,
    tile_innermost_n,
    tile_loops,
)


def softmax(xs: te.Tensor) -> tuple[te.Tensor, te.Tensor]:
    """Compute the nominator and denominator of softmax. Dividing them produces the normal softmax result.
    NOTE: only supports last axis reduction for now."""

    reduced_shape = xs.shape[:-1]
    k = te.reduce_axis((0, xs.shape[-1]), name="k")
    s_max = te.compute(
        reduced_shape, lambda *i: te.max(xs(*i, k), axis=k), name="T_softmax_maxelem"
    )
    s_exp = te.compute(
        xs.shape, lambda *axes: te.exp(xs(*axes) - s_max(*axes[:-1])), name="T_softmax_exp"
    )
    s_expsum = te.compute(
        reduced_shape, lambda *axes: te.sum(s_exp(*axes, k), axis=k), name="T_softmax_expsum"
    )
    return s_exp, s_expsum


def create_softmax(shape: tuple[int, ...]):
    xs = te.placeholder(shape, "float32", name="xs")
    s_exp, s_expsum = softmax(xs)
    s_result = te.compute(xs.shape, lambda *i: s_exp(*i) / s_expsum(*i[:-1]), name="T_softmax_norm")
    func = te.create_prim_func([xs, s_result])
    return IRModule({"softmax": func})


def schedule_softmax_0(sch: tir.Schedule):
    # One single kernel where a block fully processes each row (i.e. no tiling in the column dimension).
    # This seems to be one of the implementations used in OneFlow.
    # See https://oneflow2020.medium.com/how-to-implement-an-efficient-softmax-cuda-kernel-oneflow-performance-optimization-sharing-405ad56e9031

    bmax = sch.get_block("T_softmax_maxelem")
    bexp = sch.get_block("T_softmax_exp")
    bsum = sch.get_block("T_softmax_expsum")
    bnorm = sch.get_block("T_softmax_norm")
    *axes, i, j = sch.get_loops(bmax)
    (i0,) = tile_loops(sch, [i], [2], inner_part_factor=1)
    bind_block_idx(sch, [*axes, i0])
    sch.reverse_compute_at(bexp, i0)
    sch.reverse_compute_at(bsum, i0)
    sch.reverse_compute_at(bnorm, i0)
    for blk in bmax, bexp, bsum:
        sch.set_scope(blk, 0, "shared")
    return sch


def schedule_softmax_1(sch: tir.Schedule, inline_exp: bool = False):
    # Creates 3 separate kernels for softmax: max-reduction, shift-exp-sum, and division.
    # See the Naive kernel here: https://maharshi.bearblog.dev/optimizing-softmax-cuda/
    # (NOTE: their version has the `exp` inlined, which means the `exp` is computed twice.)

    bmax = sch.get_block("T_softmax_maxelem")
    bexp = sch.get_block("T_softmax_exp")
    bsum = sch.get_block("T_softmax_expsum")
    bnorm = sch.get_block("T_softmax_norm")

    # 1. Compute the row-wise max in a kernel.
    axes, (i0, j0) = tile_innermost_n(sch, bmax, [128, 128])
    bmax_rf = sch.rfactor(j0, factor_axis=0, merge_loops=True)
    bind_block_idx(sch, [*axes, i0])
    sch.set_scope(bmax_rf, 0, "shared")
    bmax_j0 = j0
    # The output of bmax is used by other kernels, so we need to cache-write it
    # so that a copy is written to the global memory.
    sch.reverse_compute_at(sch.cache_write(bmax, 0, "shared"), i0)
    annotate_triton_params(sch, bmax)

    # 2. Compute the subtraction, exp, and sum in a kernel.
    if inline_exp:
        sch.compute_inline(bexp)
        axes, (i0, j0) = tile_innermost_n(sch, bsum, [128, 128])
        bsum_rf = sch.rfactor(j0, factor_axis=0, merge_loops=True)
    else:
        axes, (i0, j0) = tile_innermost_n(sch, bexp, [128, 128])
        bsum_rf = sch.rolling_update(bsum, j0, factor_axis=0)
    bind_block_idx(sch, [*axes, i0])
    sch.set_scope(bsum_rf, 0, "shared")
    bsum_j0 = j0
    sch.reverse_compute_at(sch.cache_write(bsum, 0, "shared"), i0)
    if inline_exp:
        annotate_triton_params(sch, bsum)
    else:
        # `bsum_rf` should read from the cache (shared) version directly.
        sch.reverse_compute_at(sch.cache_write(bexp, 0, "shared", consumer_blocks=[bsum_rf]), j0)
        annotate_triton_params(sch, bexp)

    # 3. Compute the division in a kernel.
    axes, (i0, j0) = tile_innermost_n(sch, bnorm, [128, 128])
    bind_block_idx(sch, [*axes, i0, j0])
    annotate_triton_params(sch, bnorm)

    # Due to TVM-to-Triton requirements, we'll need to separate out the init stmt of some blocks.
    # Reduction decomposition is best done at the end of the schedule.
    sch.decompose_reduction(bmax, bmax_j0)
    sch.decompose_reduction(bsum, bsum_j0)

    return sch


def schedule_softmax_2(sch: tir.Schedule):
    # Create 2 kernels for softmax. The first kernel computes everything except the final division.
    # The second kernel computes the division.
    bmax = sch.get_block("T_softmax_maxelem")
    bsum = sch.get_block("T_softmax_expsum")
    bnorm = sch.get_block("T_softmax_norm")

    # Compute the row-wise max in a kernel.
    # We create the loop nest from the cache read block of `bmax`,
    # because it's easier to work with a spatial block.
    # For example, this allows us to use `sch.rolling_update` for bmax instead of manually rfactoring.
    bmax_cr = sch.cache_read(bmax, 0, "shared")
    axes, (i0, j0) = tile_innermost_n(sch, bmax_cr, [128, 128])
    bmax_rf = sch.rolling_update(bmax, j0, factor_axis=0)
    bind_block_idx(sch, [*axes, i0])
    sch.set_scope(bmax_rf, 0, "shared")
    sch.set_scope(bmax, 0, "shared")
    # Rolling-update `bsum` into the loop nest of `bmax`. This should also bring `bexp` into the loop nest.
    sch.rolling_update(bsum, j0, factor_axis=0)
    sch.reverse_compute_at(sch.cache_write(bsum, 0, "shared"), i0)
    # Split scan buffer, decompose reduction, and annotate triton params.
    sch.split_scan_buffer(bmax, j0, 0)
    sch.decompose_reduction(bsum, j0)
    annotate_triton_params(sch, bmax_cr)

    # Compute the division in a kernel.
    axes, (i0, j0) = tile_innermost_n(sch, bnorm, [128, 128])
    bind_block_idx(sch, [*axes, i0, j0])
    annotate_triton_params(sch, bnorm)
    return sch
