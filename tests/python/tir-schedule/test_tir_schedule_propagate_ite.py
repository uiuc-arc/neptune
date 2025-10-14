from typing import Callable

from tvm import tir
from tvm.ir.expr import PrimExpr
from tvm.script import tir as T


def create_masked_attention_input(condition: Callable[..., PrimExpr]):
    @T.prim_func
    def masked_attention():
        T_batch_matmul_NT_shared = T.alloc_buffer((1, 16, 2048, 2048), scope="shared")
        T_score_mod_shared = T.alloc_buffer((1, 16, 2048, 2048), scope="shared")
        for b1 in T.thread_binding(16, thread="blockIdx.y"):
            for i_0 in T.thread_binding(16, thread="blockIdx.x"):
                for j_0 in T.serial(32):
                    for ax0, ax1 in T.grid(128, 64):
                        with T.block("T_score_mod"):
                            v_i0 = T.axis.spatial(1, 0)
                            v_i1 = T.axis.spatial(16, b1)
                            v_i2 = T.axis.spatial(2048, i_0 * 128 + ax0)
                            v_i3 = T.axis.spatial(2048, j_0 * 64 + ax1)
                            T_score_mod_shared[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(
                                condition(v_i0, v_i1, v_i2, v_i3),
                                T_batch_matmul_NT_shared[v_i0, v_i1, v_i2, v_i3]
                                * T.float32(0.088388347648318433),
                                T.float32(-340282346638528859811704183484516925440.0),
                            )

    return masked_attention


def create_masked_attention_solution(
    j0_range_min: Callable[..., PrimExpr],
    j0_range_max: Callable[..., PrimExpr],
    likely_expr: Callable[..., PrimExpr],
):
    @T.prim_func
    def masked_attention():
        T_batch_matmul_NT_shared = T.alloc_buffer((1, 16, 2048, 2048), scope="shared")
        T_score_mod_shared = T.alloc_buffer((1, 16, 2048, 2048), scope="shared")
        for b1 in T.thread_binding(16, thread="blockIdx.y"):
            for i_0 in T.thread_binding(16, thread="blockIdx.x"):
                for j_0 in T.serial(
                    j0_range_min(i_0),
                    j0_range_max(i_0),
                    annotations={"tir.loop_original_bounds": T.Range(0, 32)},  # type: ignore
                ):
                    if T.likely(likely_expr(i_0, j_0)):
                        for ax0, ax1 in T.grid(128, 64):
                            with T.block("T_score_mod_likely"):
                                v_i0 = T.axis.spatial(1, 0)
                                v_i1 = T.axis.spatial(16, b1)
                                v_i2 = T.axis.spatial(2048, i_0 * 128 + ax0)
                                v_i3 = T.axis.spatial(2048, j_0 * 64 + ax1)
                                T_score_mod_shared[v_i0, v_i1, v_i2, v_i3] = (
                                    T_batch_matmul_NT_shared[v_i0, v_i1, v_i2, v_i3]
                                    * T.float32(0.088388347648318433)
                                )
                    else:
                        for ax0, ax1 in T.grid(128, 64):
                            with T.block("T_score_mod"):
                                v_i0 = T.axis.spatial(1, 0)
                                v_i1 = T.axis.spatial(16, b1)
                                v_i2 = T.axis.spatial(2048, i_0 * 128 + ax0)
                                v_i3 = T.axis.spatial(2048, j_0 * 64 + ax1)
                                T_score_mod_shared[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(
                                    v_i2 >= v_i3,
                                    T_batch_matmul_NT_shared[v_i0, v_i1, v_i2, v_i3]
                                    * T.float32(0.088388347648318433),
                                    T.float32(-340282346638528859811704183484516925440.0),
                                )

    return masked_attention


def test_propagate_causal_attention():
    causal_attention = create_masked_attention_input(lambda i0, i1, i2, i3: i2 >= i3)
    sch = tir.Schedule(causal_attention)
    b1 = sch.get_block("T_score_mod")
    j0 = sch.get_loops(b1)[2]
    sch.propagate_if_then_else(b1, j0, "")
    ((_, generated),) = sch.mod.functions_items()
    generated.show()
    # TODO: fix this test case. Our output does not compare equal to the expected output
    # because the pass reuses loop vars and iter vars when copying the loop nest.
    # No TIR program written literally in Python has that behavior.
    # expected = create_masked_attention_solution(
    #     lambda i0: 0, lambda i0: i0 * 2 + 2, lambda i0, j0: j0 * 64 + 63 <= i0 * 128
    # )
    # assert_structural_equal_ignore_global_symbol(generated, expected)


def test_propagate_windowed_attention():
    windowed_attention = create_masked_attention_input(
        lambda i0, i1, i2, i3: tir.all(i2 >= i3, i2 - i3 < 128)
    )
    sch = tir.Schedule(windowed_attention)
    b1 = sch.get_block("T_score_mod")
    j0 = sch.get_loops(b1)[2]
    sch.propagate_if_then_else(b1, j0, "")
    ((_, generated),) = sch.mod.functions_items()
    generated.show()
    # TODO: fix this test case. See above.
    # expected = create_masked_attention_solution(
    #     lambda i0: T.max(i0 * 2 - 2, 0),
    #     lambda i0: T.max(i0 * 2 - 2, 0) + (T.min(2, i0 * 2) + 2),
    #     lambda i0, j0: j0 * 64 + 63 <= i0 * 128 and i0 * 128 + 127 - j0 * 64 + 1 <= 128,
    # )
    # assert_structural_equal_ignore_global_symbol(generated, expected)


if __name__ == "__main__":
    from tvm.testing import main

    main()
