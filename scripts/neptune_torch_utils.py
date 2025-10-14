import torch
from torch.library import triton_op, wrap_triton
from torch.utils._triton import has_triton

import tvm
from tvm.runtime.triton import _source_str_to_triton_function
from logging import warning

from tvm.ir.transform import Sequential
from tvm.tir import transform as tr

build_triton_kernels = tvm.get_global_func("tir.transform.TritonBuildAndCollectKernel")
extract_launch_args = tvm.get_global_func("tir.transform.extract_launch_args")

# A modified versoin of scripts.felix_attn.ours.build.py::tvm_triton_build
# TODO: Move this to felix_attn.ours.build.py?
def tvm_triton_build_torch(mod, target):
    assert has_triton(), "This function requires torch to have triton support."
    if target.host is None:
        warning(f"Target {target} has no host, using LLVM as default")
        target = target.with_host("llvm")
    # These passes transform each function that has been tile-formed (has `tir.tile_expr_form` attribute).
    # Eventually TritonBuildKernel calls Triton to create (one or multiple) PTX kernels as `CUDAModule`s
    # and put them in the `external_mods` attribute of the IRModule.
    # Functions that don't have `tir.tile_expr_form` attribute are untouched.
    passes = [
        tr.LiftThreadBinding(),
        tr.ConvertBlocksToOpaque(),
        tr.LowerOpaqueBlock(),
        # Loop partitioning (for masked attns, like causal attn)
        tr.LoopPartition(),
        tr.Simplify(),
        # Build Triton kernels. BindTarget and AnnotateDeviceRegions prepare for SplitHostDevice,
        # which TritonBuildKernel calls internally.
        tr.BindTarget(target),
        tr.AnnotateDeviceRegions(),
    ]
    mod = Sequential(passes)(mod)

    launch_args = [(gv.name_hint, extract_launch_args(func)) for gv, func in mod.functions_items()]

    (names, sources) = build_triton_kernels(mod)
    assert len(names) == len(sources), "Mismatch in number of kernel names and sources."
    assert len(names) == 1, "TODO: more than one kernel not supported yet."
    triton_callable = _source_str_to_triton_function(sources[0], names[0])
    grid = [x.value for x in launch_args[0][1]]
    # TODO this captures only the grid, but not the pipeline stages or other
    # tunable args captured in the full Neptune tvm backend.
    return wrap_triton(triton_callable)[lambda meta: grid]
  
