import pytest
from tvm import tir
from tvm.ir.module import IRModule
from tvm.script import tir as T
from tvm.tir.schedule.testing import assert_structural_equal_ignore_global_symbol


@T.prim_func
def stage0(
    T_batch_matmul_NT: T.Buffer((1, 16, 1, 2048), "float16"),  # type: ignore
    v: T.Buffer((1, 16, 2048, 128), "float16"),  # type: ignore
):
    T_score_mod = T.alloc_buffer((1, 16, 1, 2048))
    T_softmax_maxelem = T.alloc_buffer((1, 16, 1))
    T_softmax_exp = T.alloc_buffer((1, 16, 1, 2048))
    T_softmax_exp_cast = T.alloc_buffer((1, 16, 1, 2048), "float16")
    T_batch_matmul_NN = T.alloc_buffer((1, 16, 1, 128))
    T_softmax_expsum = T.alloc_buffer((1, 16, 1))
    for b0_b1_fused in T.thread_binding(16, thread="blockIdx.z"):
        for j_0 in T.thread_binding(64, thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 32):
                with T.block("T_score_mod"):
                    v_i0 = T.axis.spatial(1, ax0)
                    v_i1 = T.axis.spatial(16, b0_b1_fused + ax1)
                    v_i2 = T.axis.spatial(1, ax2)
                    v_i3 = T.axis.spatial(2048, j_0 * 32 + ax3)
                    T_score_mod[v_i0, v_i1, v_i2, v_i3] = T_batch_matmul_NT[
                        v_i0, v_i1, v_i2, v_i3
                    ] * T.float32(0.088388347648318433)
    for i0, i1, i2, k_1 in T.grid(1, 16, 1, 2048):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k_1])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(
                    -340282346638528859811704183484516925440.0
                )
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(
                T_softmax_maxelem[v_i0, v_i1, v_i2], T_score_mod[v_i0, v_i1, v_i2, v_k]
            )
    for i0, i1, i2, i3 in T.grid(1, 16, 1, 2048):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(
                T_score_mod[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2]
            )
    for i0, i1, i2, i3 in T.grid(1, 16, 1, 2048):
        with T.block("T_softmax_exp_cast"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_softmax_exp_cast[v_i0, v_i1, v_i2, v_i3] = T.Cast(
                "float16", T_softmax_exp[v_i0, v_i1, v_i2, v_i3]
            )
    for b0, b1, i, j, k_1 in T.grid(1, 16, 1, 128, 2048):
        with T.block("T_batch_matmul_NN"):
            v_b0, v_b1, v_i, v_j, v_k = T.axis.remap("SSSSR", [b0, b1, i, j, k_1])
            with T.init():
                T_batch_matmul_NN[v_b0, v_b1, v_i, v_j] = T.float32(0.0)
            T_batch_matmul_NN[v_b0, v_b1, v_i, v_j] = T_batch_matmul_NN[
                v_b0, v_b1, v_i, v_j
            ] + T.Cast("float32", T_softmax_exp_cast[v_b0, v_b1, v_i, v_k]) * T.Cast(
                "float32", v[v_b0, v_b1, v_k, v_j]
            )
    for i0, i1, i2, k_1 in T.grid(1, 16, 1, 2048):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k_1])
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0.0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = (
                T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
            )


@T.prim_func
def stage1(
    T_batch_matmul_NT: T.Buffer((1, 16, 1, 2048), "float16"),  # type: ignore
    v: T.Buffer((1, 16, 2048, 128), "float16"),  # type: ignore
):
    T_score_mod = T.alloc_buffer((1, 16, 1, 2048))
    T_softmax_maxelem = T.alloc_buffer((1, 16, 1))
    T_softmax_exp = T.alloc_buffer((1, 16, 1, 2048))
    T_softmax_exp_cast = T.alloc_buffer((1, 16, 1, 2048), "float16")
    T_batch_matmul_NN = T.alloc_buffer((1, 16, 1, 128))
    T_softmax_expsum = T.alloc_buffer((1, 16, 1))
    T_softmax_maxelem_rf = T.alloc_buffer((64, 1, 16, 1))
    for b0_b1_fused in T.thread_binding(16, thread="blockIdx.z"):
        for j_0 in T.thread_binding(64, thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 32):
                with T.block("T_score_mod"):
                    v_i0 = T.axis.spatial(1, ax0)
                    v_i1 = T.axis.spatial(16, b0_b1_fused + ax1)
                    v_i2 = T.axis.spatial(1, ax2)
                    v_i3 = T.axis.spatial(2048, j_0 * 32 + ax3)
                    T_score_mod[v_i0, v_i1, v_i2, v_i3] = T.Cast(
                        "float32", T_batch_matmul_NT[v_i0, v_i1, v_i2, v_i3]
                    ) * T.float32(0.088388347648318433)
            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 32):
                with T.block("T_softmax_maxelem_rf"):
                    vj_0, v_i0 = T.axis.remap("SS", [j_0, ax0])
                    v_i1 = T.axis.spatial(16, b0_b1_fused + ax1)
                    v_i2, vax3 = T.axis.remap("SR", [ax2, ax3])
                    T.block_attr({"tir.rfactor_data": [T.bool(False), T_softmax_maxelem, 0]})
                    with T.init():
                        T_softmax_maxelem_rf[vj_0, v_i0, v_i1, v_i2] = T.float32(
                            -340282346638528859811704183484516925440.0
                        )
                    T_softmax_maxelem_rf[vj_0, v_i0, v_i1, v_i2] = T.max(
                        T_softmax_maxelem_rf[vj_0, v_i0, v_i1, v_i2],
                        T_score_mod[v_i0, v_i1, v_i2, vj_0 * 32 + vax3],
                    )
    for ax0, ax1, ax2, ax3 in T.grid(64, 1, 16, 1):
        with T.block("T_softmax_maxelem"):
            vj_0, v_i0, v_i1, v_i2 = T.axis.remap("RSSS", [ax0, ax1, ax2, ax3])
            T.block_attr({"tir.rfactor_data": [T.bool(True), T_softmax_maxelem_rf, 0]})
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(
                    -340282346638528859811704183484516925440.0
                )
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(
                T_softmax_maxelem[v_i0, v_i1, v_i2], T_softmax_maxelem_rf[vj_0, v_i0, v_i1, v_i2]
            )
    for i0, i1, i2, i3 in T.grid(1, 16, 1, 2048):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(
                T_score_mod[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2]
            )
    for i0, i1, i2, i3 in T.grid(1, 16, 1, 2048):
        with T.block("T_softmax_exp_cast"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_softmax_exp_cast[v_i0, v_i1, v_i2, v_i3] = T.Cast(
                "float16", T_softmax_exp[v_i0, v_i1, v_i2, v_i3]
            )
    for b0, b1, i, j, k_1 in T.grid(1, 16, 1, 128, 2048):
        with T.block("T_batch_matmul_NN"):
            v_b0, v_b1, v_i, v_j, v_k = T.axis.remap("SSSSR", [b0, b1, i, j, k_1])
            with T.init():
                T_batch_matmul_NN[v_b0, v_b1, v_i, v_j] = T.float32(0.0)
            T_batch_matmul_NN[v_b0, v_b1, v_i, v_j] = T_batch_matmul_NN[
                v_b0, v_b1, v_i, v_j
            ] + T.Cast("float32", T_softmax_exp_cast[v_b0, v_b1, v_i, v_k]) * T.Cast(
                "float32", v[v_b0, v_b1, v_k, v_j]
            )
    for i0, i1, i2, k_1 in T.grid(1, 16, 1, 2048):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k_1])
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0.0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = (
                T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
            )


@T.prim_func
def stage2(
    T_batch_matmul_NT: T.Buffer((1, 16, 1, 2048), "float16"),  # type: ignore
    v: T.Buffer((1, 16, 2048, 128), "float16"),  # type: ignore
):
    T_score_mod = T.alloc_buffer((1, 16, 1, 2048))
    T_softmax_maxelem = T.alloc_buffer((1, 16, 1))
    T_softmax_exp = T.alloc_buffer((1, 16, 1, 2048))
    T_softmax_exp_cast = T.alloc_buffer((1, 16, 1, 2048), "float16")
    T_batch_matmul_NN = T.alloc_buffer((1, 16, 1, 128))
    T_softmax_expsum = T.alloc_buffer((1, 16, 1))
    T_softmax_maxelem_rf = T.alloc_buffer((64, 1, 16, 1))
    T_softmax_expsum_rf = T.alloc_buffer((64, 1, 16, 1))
    for b0_b1_fused in T.thread_binding(16, thread="blockIdx.z"):
        for j_0 in T.thread_binding(64, thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 32):
                with T.block("T_score_mod"):
                    v_i0 = T.axis.spatial(1, ax0)
                    v_i1 = T.axis.spatial(16, b0_b1_fused + ax1)
                    v_i2 = T.axis.spatial(1, ax2)
                    v_i3 = T.axis.spatial(2048, j_0 * 32 + ax3)
                    T_score_mod[v_i0, v_i1, v_i2, v_i3] = T.Cast(
                        "float32", T_batch_matmul_NT[v_i0, v_i1, v_i2, v_i3]
                    ) * T.float32(0.088388347648318433)
            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 32):
                with T.block("T_softmax_maxelem_rf"):
                    vj_0, v_i0 = T.axis.remap("SS", [j_0, ax0])
                    v_i1 = T.axis.spatial(16, b0_b1_fused + ax1)
                    v_i2, vax3 = T.axis.remap("SR", [ax2, ax3])
                    T.block_attr({"tir.rfactor_data": [T.bool(False), T_softmax_maxelem, 0]})
                    with T.init():
                        T_softmax_maxelem_rf[vj_0, v_i0, v_i1, v_i2] = T.float32(
                            -340282346638528859811704183484516925440.0
                        )
                    T_softmax_maxelem_rf[vj_0, v_i0, v_i1, v_i2] = T.max(
                        T_softmax_maxelem_rf[vj_0, v_i0, v_i1, v_i2],
                        T_score_mod[v_i0, v_i1, v_i2, vj_0 * 32 + vax3],
                    )
            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 32):
                with T.block("T_softmax_exp"):
                    v_i0 = T.axis.spatial(1, ax0)
                    v_i1 = T.axis.spatial(16, b0_b1_fused + ax1)
                    v_i2, v_i3_0, v_i3_1 = T.axis.remap("SSS", [ax2, j_0, ax3])
                    T_softmax_exp[v_i0, v_i1, v_i2, v_i3_0 * 32 + v_i3_1] = T.exp(
                        T_score_mod[v_i0, v_i1, v_i2, v_i3_0 * 32 + v_i3_1]
                        - T_softmax_maxelem_rf[v_i3_0, v_i0, v_i1, v_i2]
                    )
            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 32):
                with T.block("T_softmax_expsum_rf"):
                    vj_0, v_i0 = T.axis.remap("SS", [j_0, ax0])
                    v_i1 = T.axis.spatial(16, b0_b1_fused + ax1)
                    v_i2, vax3 = T.axis.remap("SR", [ax2, ax3])
                    T.block_attr({"tir.rfactor_data": [T.bool(False), T_softmax_expsum, 0]})
                    with T.init():
                        T_softmax_expsum_rf[vj_0, v_i0, v_i1, v_i2] = T.float32(0.0)
                    T_softmax_expsum_rf[vj_0, v_i0, v_i1, v_i2] = (
                        T_softmax_expsum_rf[vj_0, v_i0, v_i1, v_i2]
                        + T_softmax_exp[v_i0, v_i1, v_i2, vj_0 * 32 + vax3]
                    )
    for ax0, ax1, ax2, ax3 in T.grid(64, 1, 16, 1):
        with T.block("T_softmax_maxelem"):
            vj_0, v_i0, v_i1, v_i2 = T.axis.remap("RSSS", [ax0, ax1, ax2, ax3])
            T.block_attr({"tir.rfactor_data": [T.bool(True), T_softmax_maxelem_rf, 0]})
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(
                    -340282346638528859811704183484516925440.0
                )
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(
                T_softmax_maxelem[v_i0, v_i1, v_i2], T_softmax_maxelem_rf[vj_0, v_i0, v_i1, v_i2]
            )
    for i0, i1, i2, i3 in T.grid(1, 16, 1, 2048):
        with T.block("T_softmax_exp_cast"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_softmax_exp_cast[v_i0, v_i1, v_i2, v_i3] = T.Cast(
                "float16", T_softmax_exp[v_i0, v_i1, v_i2, v_i3]
            )
    for b0, b1, i, j, k_1 in T.grid(1, 16, 1, 128, 2048):
        with T.block("T_batch_matmul_NN"):
            v_b0, v_b1, v_i, v_j, v_k = T.axis.remap("SSSSR", [b0, b1, i, j, k_1])
            with T.init():
                T_batch_matmul_NN[v_b0, v_b1, v_i, v_j] = T.float32(0.0)
            T_batch_matmul_NN[v_b0, v_b1, v_i, v_j] = T_batch_matmul_NN[
                v_b0, v_b1, v_i, v_j
            ] + T.Cast("float32", T_softmax_exp_cast[v_b0, v_b1, v_i, v_k]) * T.Cast(
                "float32", v[v_b0, v_b1, v_k, v_j]
            )
    for ax0, ax1, ax2, ax3 in T.grid(64, 1, 16, 1):
        with T.block("T_softmax_expsum"):
            vj_0, v_i0, v_i1, v_i2 = T.axis.remap("RSSS", [ax0, ax1, ax2, ax3])
            T.block_attr({"tir.rfactor_data": [T.bool(True), T_softmax_expsum_rf, 0]})
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0.0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[
                v_i0, v_i1, v_i2
            ] + T_softmax_expsum_rf[vj_0, v_i0, v_i1, v_i2] * T.exp(
                T_softmax_maxelem_rf[vj_0, 0, v_i1, 0] - T_softmax_maxelem[0, v_i1, 0]
            )


@T.prim_func
def stage3(
    T_batch_matmul_NT: T.Buffer((1, 16, 1, 2048), "float16"),  # type: ignore
    v: T.Buffer((1, 16, 2048, 128), "float16"),  # type: ignore
):
    T_score_mod = T.alloc_buffer((1, 16, 1, 2048))
    T_softmax_maxelem = T.alloc_buffer((1, 16, 1))
    T_softmax_exp = T.alloc_buffer((1, 16, 1, 2048))
    T_softmax_exp_cast = T.alloc_buffer((1, 16, 1, 2048), "float16")
    T_batch_matmul_NN = T.alloc_buffer((1, 16, 1, 128))
    T_softmax_expsum = T.alloc_buffer((1, 16, 1))
    T_softmax_maxelem_rf = T.alloc_buffer((64, 1, 16, 1))
    T_softmax_expsum_rf = T.alloc_buffer((64, 1, 16, 1))
    T_batch_matmul_NN_rf = T.alloc_buffer((64, 1, 16, 1, 128))
    for b0_b1_fused in T.thread_binding(16, thread="blockIdx.z"):
        for j_0 in T.thread_binding(64, thread="blockIdx.x"):
            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 32):
                with T.block("T_score_mod"):
                    v_i0 = T.axis.spatial(1, ax0)
                    v_i1 = T.axis.spatial(16, b0_b1_fused + ax1)
                    v_i2 = T.axis.spatial(1, ax2)
                    v_i3 = T.axis.spatial(2048, j_0 * 32 + ax3)
                    T_score_mod[v_i0, v_i1, v_i2, v_i3] = T.Cast(
                        "float32", T_batch_matmul_NT[v_i0, v_i1, v_i2, v_i3]
                    ) * T.float32(0.088388347648318433)
            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 32):
                with T.block("T_softmax_maxelem_rf"):
                    vj_0, v_i0 = T.axis.remap("SS", [j_0, ax0])
                    v_i1 = T.axis.spatial(16, b0_b1_fused + ax1)
                    v_i2, vax3 = T.axis.remap("SR", [ax2, ax3])
                    T.block_attr({"tir.rfactor_data": [T.bool(False), T_softmax_maxelem, 0]})
                    with T.init():
                        T_softmax_maxelem_rf[vj_0, v_i0, v_i1, v_i2] = T.float32(
                            -340282346638528859811704183484516925440.0
                        )
                    T_softmax_maxelem_rf[vj_0, v_i0, v_i1, v_i2] = T.max(
                        T_softmax_maxelem_rf[vj_0, v_i0, v_i1, v_i2],
                        T_score_mod[v_i0, v_i1, v_i2, vj_0 * 32 + vax3],
                    )
            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 32):
                with T.block("T_softmax_exp"):
                    v_i0 = T.axis.spatial(1, ax0)
                    v_i1 = T.axis.spatial(16, b0_b1_fused + ax1)
                    v_i2, v_i3_0, v_i3_1 = T.axis.remap("SSS", [ax2, j_0, ax3])
                    T_softmax_exp[v_i0, v_i1, v_i2, v_i3_0 * 32 + v_i3_1] = T.exp(
                        T_score_mod[v_i0, v_i1, v_i2, v_i3_0 * 32 + v_i3_1]
                        - T_softmax_maxelem_rf[v_i3_0, v_i0, v_i1, v_i2]
                    )
            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 32):
                with T.block("T_softmax_expsum_rf"):
                    vj_0, v_i0 = T.axis.remap("SS", [j_0, ax0])
                    v_i1 = T.axis.spatial(16, b0_b1_fused + ax1)
                    v_i2, vax3 = T.axis.remap("SR", [ax2, ax3])
                    T.block_attr({"tir.rfactor_data": [T.bool(False), T_softmax_expsum, 0]})
                    with T.init():
                        T_softmax_expsum_rf[vj_0, v_i0, v_i1, v_i2] = T.float32(0.0)
                    T_softmax_expsum_rf[vj_0, v_i0, v_i1, v_i2] = (
                        T_softmax_expsum_rf[vj_0, v_i0, v_i1, v_i2]
                        + T_softmax_exp[v_i0, v_i1, v_i2, vj_0 * 32 + vax3]
                    )
            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 32):
                with T.block("T_softmax_exp_cast"):
                    v_i0 = T.axis.spatial(1, ax0)
                    v_i1 = T.axis.spatial(16, b0_b1_fused + ax1)
                    v_i2 = T.axis.spatial(1, ax2)
                    v_i3 = T.axis.spatial(2048, j_0 * 32 + ax3)
                    T_softmax_exp_cast[v_i0, v_i1, v_i2, v_i3] = T.Cast(
                        "float16", T_softmax_exp[v_i0, v_i1, v_i2, v_i3]
                    )
            for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 1, 1, 128, 32):
                with T.block("T_batch_matmul_NN_rf"):
                    vj_0, v_b0 = T.axis.remap("SS", [j_0, ax0])
                    v_b1 = T.axis.spatial(16, b0_b1_fused + ax1)
                    v_i, v_j, vax4 = T.axis.remap("SSR", [ax2, ax3, ax4])
                    T.block_attr({"tir.rfactor_data": [T.bool(False), T_batch_matmul_NN, 0]})
                    with T.init():
                        T_batch_matmul_NN_rf[vj_0, v_b0, v_b1, v_i, v_j] = T.float32(0.0)
                    T_batch_matmul_NN_rf[vj_0, v_b0, v_b1, v_i, v_j] = T_batch_matmul_NN_rf[
                        vj_0, v_b0, v_b1, v_i, v_j
                    ] + T.Cast(
                        "float32", T_softmax_exp_cast[v_b0, v_b1, v_i, vj_0 * 32 + vax4]
                    ) * T.Cast("float32", v[v_b0, v_b1, vj_0 * 32 + vax4, v_j])
    for ax0, ax1, ax2, ax3 in T.grid(64, 1, 16, 1):
        with T.block("T_softmax_maxelem"):
            vj_0, v_i0, v_i1, v_i2 = T.axis.remap("RSSS", [ax0, ax1, ax2, ax3])
            T.block_attr({"tir.rfactor_data": [T.bool(True), T_softmax_maxelem_rf, 0]})
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(
                    -340282346638528859811704183484516925440.0
                )
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(
                T_softmax_maxelem[v_i0, v_i1, v_i2], T_softmax_maxelem_rf[vj_0, v_i0, v_i1, v_i2]
            )
    for ax0, ax1, ax2, ax3 in T.grid(64, 1, 16, 1):
        with T.block("T_softmax_expsum"):
            vj_0, v_i0, v_i1, v_i2 = T.axis.remap("RSSS", [ax0, ax1, ax2, ax3])
            T.block_attr({"tir.rfactor_data": [T.bool(True), T_softmax_expsum_rf, 0]})
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0.0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[
                v_i0, v_i1, v_i2
            ] + T_softmax_expsum_rf[vj_0, v_i0, v_i1, v_i2] * T.exp(
                T_softmax_maxelem_rf[vj_0, 0, v_i1, 0] - T_softmax_maxelem[0, v_i1, 0]
            )
    for ax0, ax1, ax2, ax3, ax4 in T.grid(64, 1, 16, 1, 128):
        with T.block("T_batch_matmul_NN"):
            vj_0, v_b0, v_b1, v_i, v_j = T.axis.remap("RSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.block_attr({"tir.rfactor_data": [T.bool(True), T_batch_matmul_NN_rf, 0]})
            with T.init():
                T_batch_matmul_NN[v_b0, v_b1, v_i, v_j] = T.float32(0.0)
            T_batch_matmul_NN[v_b0, v_b1, v_i, v_j] = T_batch_matmul_NN[
                v_b0, v_b1, v_i, v_j
            ] + T_batch_matmul_NN_rf[vj_0, v_b0, v_b1, v_i, v_j] * T.exp(
                T_softmax_maxelem_rf[vj_0, 0, v_b1, 0] - T_softmax_maxelem[0, v_b1, 0]
            )


@pytest.mark.parametrize(
    "input, apply_to, expected",
    [
        (stage0, "T_softmax_maxelem", stage1),
        (stage1, "T_softmax_expsum", stage2),
        (stage2, "T_batch_matmul_NN", stage3),
    ],
)
def test_flashattn_roll_update_steps(input, apply_to, expected):
    module = IRModule.from_expr(input)
    sch = tir.Schedule(module)
    j0 = sch.get_loops("T_score_mod")[1]
    sch.split_k_update(apply_to, j0, 0)
    ((_, generated),) = sch.mod.functions_items()
    assert_structural_equal_ignore_global_symbol(generated, expected)


if __name__ == "__main__":
    from tvm.testing import main

    main()
