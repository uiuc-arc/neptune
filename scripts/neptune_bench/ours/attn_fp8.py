import numpy as np
from tvm import ir, te, tir

from ..operators import BenchmarkOperator
from .attn_plain import general_attention, schedule_attention_plain, schedule_full_attn_flash
from .runner import NeptuneRunner


def create_attention_fp8(
    input_shape: tuple[int, int, int, int], q_fp8: bool, func_name: str = "flash_attention_fp8"
):
    """Create an attention kernel with FP8 inputs. K and V are always FP8,
    while Q can be FP8 or FP16."""
    head_dim = input_shape[3]
    q_dtype = "e5m2_float8" if q_fp8 else "float16"
    q = te.placeholder(input_shape, q_dtype, name="q")
    k = te.placeholder(input_shape, "e5m2_float8", name="k")
    v = te.placeholder(input_shape, "e5m2_float8", name="v")
    k_scale = tir.Var("k_scale", "float32")
    v_scale = tir.Var("v_scale", "float32")
    k_scale_ = k_scale / np.sqrt(head_dim)
    out = general_attention(
        q, k, v, lambda score, *args: score * k_scale_, lambda *_: True, v_scale
    )
    func = te.create_prim_func([q, k, v, k_scale, v_scale, out])
    return ir.IRModule({func_name: func})


class NeptuneFP8PrefillRunner(NeptuneRunner):
    @classmethod
    def create_from_schedulers(  # type: ignore
        cls,
        operator: BenchmarkOperator,
        q_fp8: bool,
        schedulers: tuple = (schedule_attention_plain, schedule_full_attn_flash),
    ):
        return super()._create_from_schedulers(
            operator, lambda shape: create_attention_fp8(shape, q_fp8), schedulers
        )

    def extract_shape(self, inputs) -> tuple:
        return inputs[0].shape
