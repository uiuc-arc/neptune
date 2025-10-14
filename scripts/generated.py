import torch
import triton
import triton.language as tl
from felix_attn.flashinfer import attention_fp8_paged


@triton.jit
def flash_attention_fp8_kernel(q_1, k, v, k_scale, v_scale, cast):
    prange_1 = tl.arange(0, 32)
    prange = tl.arange(0, 128)
    blockIdx_z = tl.program_id(2)
    blockIdx_y = tl.program_id(1)
    blockIdx_x = tl.program_id(0)
    q = tl.load(
        q_1
        + blockIdx_y * 262144
        + (blockIdx_x * 128 * 128 + prange[:, None] * 128)
        + prange[None, :]
    )
    softmax_maxelem_prev = tl.zeros([128], tl.float32) + -340282346638528859811704183484516925440.0
    softmax_expsum_prev = tl.zeros([128], tl.float32)
    batch_matmul_NN = tl.zeros([128, 128], tl.float32)
    for j_0 in range(0, 64):
        batch_matmul_NT = tl.dot(
            q,
            tl.cast(
                tl.load(
                    tl.make_block_ptr(
                        k + blockIdx_y * 262144,
                        shape=[128, 2048],
                        strides=[1, 128],
                        offsets=[0, j_0 * 32],
                        block_shape=[128, 32],
                        order=[1, 0],
                    )
                ),
                tl.float16,
            ),
        )
        score_mod = batch_matmul_NT * (k_scale * 0.088388347648318433)
        softmax_maxelem_rf = tl.max(score_mod, 1)
        softmax_maxelem_next = tl.maximum(softmax_maxelem_prev, softmax_maxelem_rf)
        softmax_exp = tl.exp(score_mod - softmax_maxelem_next[:, None])
        softmax_expsum_rf = tl.sum(softmax_exp, 1)
        softmax_expsum_next = (
            softmax_expsum_prev * tl.exp(softmax_maxelem_prev - softmax_maxelem_next)
            + softmax_expsum_rf
        )
        softmax_exp_cast = tl.cast(softmax_exp, tl.float16)
        batch_matmul_NN_rf = tl.dot(
            softmax_exp_cast,
            tl.cast(
                tl.load(
                    v
                    + blockIdx_y * 262144
                    + (j_0 * 32 * 128 + prange_1[:, None] * 128)
                    + prange[None, :]
                ),
                tl.float16,
            ),
        )
        batch_matmul_NN = batch_matmul_NN + batch_matmul_NN_rf
        softmax_maxelem_prev = softmax_maxelem_next
        softmax_expsum_prev = softmax_expsum_next
    softmax_norm = batch_matmul_NN / softmax_expsum_prev[:, None]
    tl.store(
        cast
        + blockIdx_y * 262144
        + (blockIdx_x * 128 * 128 + prange[:, None] * 128)
        + prange[None, :],
        softmax_norm * v_scale,
    )


def test_correctness():
    Z, H, N_CTX, HEAD_DIM = 1, 16, 2048, 128
    DEVICE = "cuda"
    q_dtype = torch.float16
    kv_dtype = torch.float8_e5m2
    torch.manual_seed(20)

    q = torch.rand((Z, H, N_CTX, HEAD_DIM)).to(q_dtype).to(DEVICE)
    k = torch.rand((Z, H, N_CTX, HEAD_DIM)).to(kv_dtype).to(DEVICE)
    v = torch.rand((Z, H, N_CTX, HEAD_DIM)).to(kv_dtype).to(DEVICE)
    k_scale = torch.rand(()).item()
    v_scale = torch.rand(()).item()
    # ref implementation
    ref_out = attention_fp8_paged(q, k, v, k_scale, v_scale, False)
    # our implementation
    out = torch.zeros(Z, H, N_CTX, HEAD_DIM, device=DEVICE, dtype=q_dtype)
    flash_attention_fp8_kernel[(triton.cdiv(N_CTX, 128), H)](
        q, k, v, k_scale, v_scale, out, num_warps=4, num_stages=1
    )
    if not torch.allclose(ref_out, out, atol=1e-2, rtol=0):
        # Find a few examples of mismatches
        mismatch = torch.abs(ref_out - out) > 1e-2
        mismatch_indices = torch.nonzero(mismatch)[:5]  # Get up to 5 examples
        print("Mismatches found. Examples:")
        for idx in mismatch_indices:
            b, h, i, j = idx
            print(f"At position [batch={b}, head={h}, i={i}, j={j}]:")
            print(f"  Reference: {ref_out[b, h, i, j]}")
            print(f"  Output:    {out[b, h, i, j]}")
            print(f"  Diff:      {abs(ref_out[b, h, i, j] - out[b, h, i, j])}")
        assert False, "Outputs don't match within tolerance"
    print("Correctness test passed")


if __name__ == "__main__":
    test_correctness()
