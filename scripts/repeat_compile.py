import shutil

from felix_attn.ours import create_general_attention, schedule_full_attn_flash
from felix_attn.ours.build import tvm_triton_build
from felix_attn.utils import get_current_gpu_info
from tvm import tir

target = get_current_gpu_info().tvm_target
mod1 = create_general_attention((1, 32, 128, 128, 128))
mod2 = create_general_attention((1, 32, 32768, 32768, 128))


@profile
def compile_loop1():
    for i in range(20):
        sch = tir.Schedule(mod1)
        sch = schedule_full_attn_flash(sch)
        shutil.rmtree("/home/yifan/.triton/cache")
        prog = tvm_triton_build(sch.mod, target)
    for i in range(20):
        sch = tir.Schedule(mod2)
        sch = schedule_full_attn_flash(sch)
        shutil.rmtree("/home/yifan/.triton/cache")
        prog = tvm_triton_build(sch.mod, target)


compile_loop1()
