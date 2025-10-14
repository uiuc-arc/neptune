import numpy as np
import torch
import tvm
from tvm import relax, te, tir
from tvm.script import relax as R

from ..runner import BenchmarkOperator, BenchmarkRunner
from ..utils import get_current_gpu_info

# Import it to register the pipeline
from .pipeline import _pipeline  # noqa: F401


def get_attention_module(
    q_shape,
    k_shape,
    v_shape,
    is_causal: bool,
    window_size: int | None = None,
    input_dt="float16",
    compute_dt="float32",
):
    sqrt_d = np.sqrt(q_shape[-1])
    neg_inf_val = tir.min_value(compute_dt)
    q_heads, k_heads = q_shape[1], k_shape[1]
    groups = q_heads // k_heads

    def causal_mask_te(score: te.Tensor):
        return te.compute(
            score.shape, lambda b, h, i, j: tir.Select(i >= j, score[b, h, i, j], neg_inf_val)
        )

    def window_mask_te(score: te.Tensor):
        return te.compute(
            score.shape,
            lambda b, h, i, j: tir.Select(
                tir.all(i >= j, i - j < window_size), score[b, h, i, j], neg_inf_val
            ),
        )

    bb = relax.BlockBuilder()
    q = relax.Var("q", R.Tensor(q_shape, input_dt))
    k = relax.Var("k", R.Tensor(k_shape, input_dt))
    v = relax.Var("v", R.Tensor(v_shape, input_dt))
    with bb.function("main", [q, k, v]):
        with bb.dataflow():
            if groups > 1:
                k = R.repeat(k, groups, 1)
                v = R.repeat(v, groups, 1)
            qkt = bb.emit(
                R.matmul(q, R.permute_dims(k, [0, 1, 3, 2]), compute_dt)
                / R.const(sqrt_d, compute_dt)
            )
            if window_size is not None:
                masked = bb.emit_te(window_mask_te, qkt)
            elif is_causal:
                masked = bb.emit_te(causal_mask_te, qkt)
            else:
                masked = qkt
            score = bb.emit(R.astype(R.nn.softmax(masked, axis=-1), input_dt))
            output = bb.emit(R.astype(R.matmul(score, v, compute_dt), input_dt))
            output = bb.emit_output(output)
        bb.emit_func_output(output)
    return bb.get()


class TVMPrefillRunner(BenchmarkRunner):
    def __init__(self, operator: BenchmarkOperator, is_causal: bool):
        super().__init__(operator, "tvm", language="tvm")
        self.is_causal = is_causal

    def setup(self, *inputs) -> bool:
        q, k, v = inputs
        # Plain Attention cannot handle too large inputs, because the score tensor gets too large.
        score_shape = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
        if np.prod(score_shape) >= 2**32:  # 1 billion elements, 4GB
            return False
        mod = get_attention_module(q.shape, k.shape, v.shape, self.is_causal)
        target = get_current_gpu_info().tvm_target
        with tvm.transform.PassContext(config={"cuda.kernels_output_dir": "/tmp/tvm_kernels"}):
            exe = relax.build(mod, target, pipeline=relax.get_pipeline("opt_llm"))
        device = tvm.device("cuda", 0)
        self.vm = relax.VirtualMachine(exe, device)
        # TODO: tvm.nd.array(torch_array, device) segfaults for arrays that are roughly larger than 1GB.
        self.q, self.k, self.v = [tvm.nd.array(data.cpu().numpy(), device) for data in [q, k, v]]
        return True

    def run(self):
        with self.mark_ctx():
            self.output = self.vm["main"](self.q, self.k, self.v)

    def get_output(self) -> torch.Tensor:
        return torch.from_numpy(self.output.numpy())


class TVMDecodeRunner(BenchmarkRunner):
    def __init__(self, operator: BenchmarkOperator):
        super().__init__(operator, "tvm", language="tvm")

    def setup(self, *inputs) -> bool:
        q, k, v = inputs
        mod = get_attention_module(q.shape, k.shape, v.shape, is_causal=False)
        target = get_current_gpu_info().tvm_target
        exe = relax.build(mod, target, pipeline=relax.get_pipeline("opt_llm"))
        device = tvm.device("cuda", 0)
        self.vm = relax.VirtualMachine(exe, device)
        self.q, self.k, self.v = [tvm.nd.array(data.cpu().numpy(), device) for data in [q, k, v]]
        return True

    def run(self):
        with self.mark_ctx():
            self.output = self.vm["main"](self.q, self.k, self.v)

    def get_output(self) -> torch.Tensor:
        return torch.from_numpy(self.output.numpy())
