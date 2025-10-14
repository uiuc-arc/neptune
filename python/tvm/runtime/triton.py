# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Triton kernel integration with TIR"""

import logging
from functools import partial

from tvm import tir
from tvm._ffi import register_func
from tvm._ffi.runtime_ctypes import DataType
from tvm.runtime import Module, create_cuda_module, create_rocm_module
from tvm.tir.expr import convert
from tvm.topi.utils import get_const_int

logger = logging.getLogger(__name__)


def _tvm_type_to_triton(tvm_type: str):
    import torch
    from triton.runtime.jit import type_canonicalisation_dict

    torch_dtype = DataType.to_torch_dtype(tvm_type, torch)
    tl_recognized_dtype = str(torch_dtype).removeprefix("torch.")
    return type_canonicalisation_dict[tl_recognized_dtype]


def _derive_triton_signature_from_params(
    triton_jit_fn, tir_params: list[tir.Buffer | tir.PrimExpr]
):
    from triton.backends.compiler import AttrsDescriptor
    from triton.runtime.jit import KernelParam

    triton_params: list[KernelParam] = triton_jit_fn.params
    assert len(triton_params) == len(tir_params), (
        f"Number of arguments does not match. Expected {len(triton_params)}, got {len(tir_params)}"
    )
    signature: dict[str, str] = {}
    pointers_indices = []
    param_types: list[str] = []
    for i, (trp, tip) in enumerate(zip(triton_params, tir_params)):
        name = trp.name
        assert not trp.is_constexpr, "Triton constexpr is not supported"
        if isinstance(tip, tir.Buffer):
            signature[name] = "*" + _tvm_type_to_triton(tip.dtype)
            pointers_indices.append(i)
            param_types.append("handle")
        else:
            assert isinstance(tip, tir.PrimExpr) and tip.dtype != "handle"
            signature[name] = _tvm_type_to_triton(tip.dtype)
            param_types.append(tip.dtype)
    attrs = AttrsDescriptor.from_hints({idx: 16 for idx in pointers_indices})  # type: ignore
    return signature, attrs, param_types


def _derive_triton_signature_from_call_args(triton_jit_fn, call_args: list[tir.PrimExpr]):
    from triton.backends.compiler import AttrsDescriptor
    from triton.runtime.jit import KernelParam

    kernel_params: list[KernelParam] = triton_jit_fn.params
    assert len(kernel_params) == len(call_args), (
        f"Number of arguments does not match, expected {len(kernel_params)}, got {len(call_args)}"
    )
    signature: dict[str, str] = {}
    constants = {}
    kernel_args = []  # Arguments to invoke the kernel
    pointers_indices = []
    for i, arg in enumerate(call_args):
        name = kernel_params[i].name
        if kernel_params[i].is_constexpr:
            constants[name] = get_const_int(arg)
            continue
        if arg.dtype == "handle":
            assert isinstance(arg, tir.Var)
            elem_type = arg.type_annotation.element_type.dtype
            pointer_type = "*" + _tvm_type_to_triton(elem_type)
            signature[name] = pointer_type
            pointers_indices.append(i)
        else:
            signature[name] = _tvm_type_to_triton(arg.dtype)
        kernel_args.append(arg)

    attrs = AttrsDescriptor.from_hints({idx: 16 for idx in pointers_indices})  # type: ignore
    return signature, constants, attrs, kernel_args


def _compile_triton_kernel(triton_jit_fn, signature, constants, attrs, compile_options):
    import triton

    source = triton.compiler.ASTSource(triton_jit_fn, signature, constants, attrs)
    triton_kernel = triton.compiler.compile(source, options=compile_options)
    kernel_metadata = triton_kernel.metadata
    assert kernel_metadata.num_ctas == 1, "Cluster is not supported"
    # NOTE: Triton API has this weird behavior where if you provide num_warps,
    # `kernel_metadata.num_warps` will become an empty dictionary.
    if "num_warps" in compile_options:
        num_warps = compile_options["num_warps"]
    else:
        num_warps = kernel_metadata.num_warps
    num_warps = int(num_warps)
    backend = kernel_metadata.target.backend
    asm = triton_kernel.asm
    fn_name = source.name
    if backend == "cuda":
        create_gpu_module = partial(create_cuda_module, asm["ptx"], kernel_name=fn_name)
    elif backend == "hip":
        create_gpu_module = partial(create_rocm_module, asm["hsaco"], kernel_name=fn_name)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return fn_name, create_gpu_module, num_warps, kernel_metadata.shared


def _derive_launch_args(
    launch_args: list[int | tir.PrimExpr], num_warps: int, shared_mem_size: int
):
    grid = launch_args
    assert len(grid) <= 3, "Triton only supports up to 3D grid"
    launch_param_tags = ["threadIdx.x"] + ["blockIdx.x", "blockIdx.y", "blockIdx.z"][: len(grid)]
    launch_args = [num_warps * 32] + list(grid)
    if shared_mem_size > 0:
        # Add shared memory size to the launch arguments
        launch_param_tags.append("tir.use_dyn_shared_memory")
        launch_args.append(shared_mem_size)
    return launch_args, launch_param_tags


def compile_triton_func_to_device_module(
    triton_jit_fn, launch_args: list[int | tir.PrimExpr], *args, **kwargs
) -> tuple[str, Module, list]:
    """Compile the kernel to a device module.

    Parameters
    ----------
    triton_jit_fn: triton.JITFunction
        The Triton kernel function.

    launch_args : List[int]
        The grid size of the kernel. A list of one to three expressions, representing the number
        of
        "blockIdx.x", "blockIdx.y", and "blockIdx.z" respectively.

    args : List[Any]
        Arguments to the kernel function.

    kwargs : Dict[str, Any]
        Additional options for the kernel compilation.
    """
    sig, consts, attrs, kargs = _derive_triton_signature_from_call_args(triton_jit_fn, list(args))
    karg_types = [arg.dtype for arg in kargs]
    fn_name, create_gpu_module, num_warps, shared_mem_size = _compile_triton_kernel(
        triton_jit_fn, sig, consts, attrs, kwargs
    )
    launch_args, launch_param_tags = _derive_launch_args(launch_args, num_warps, shared_mem_size)
    kernel_module = create_gpu_module(karg_types, launch_param_tags)
    return fn_name, kernel_module, kargs + launch_args


def _source_str_to_triton_function(source: str, fn_name: str):
    import linecache

    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    # Pick a filename that is not a real file, and insert the source into the cache,
    # so that inspect.getsource() will work.
    module_name = "__tvm_py_source_to_function"
    filename = f"{module_name}.py"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    code = compile(source, filename, "exec")
    namespace = {"tl": tl, "libdevice": libdevice}
    exec(code, namespace)
    return triton.JITFunction(namespace[fn_name])


@register_func("runtime.triton.compile_triton_source_to_device_module")
def compile_triton_source_to_device_module(
    source: str,
    func_name: str,
    launch_args: list[int | tir.PrimExpr],
    tir_params: list[tir.Buffer | tir.PrimExpr],
    num_warps: int | None,
    num_stages: int | None,
):
    """Compile a Triton kernel from Python source string to a TVM CUDAModule.
    It differs from `compile_triton_func_to_device_module` above in that
    (1) it starts from a Python source string instead of a JITFunction, and
    (2) it requires only the parameters of the TIR function to be compiled, not the arguments at the call site.
    """
    import os

    if os.environ.get("NEPTUNE_PRINT_TRITON_SOURCE", "0") == "1":
        print(source)
        print(f"{num_warps=}, {num_stages=}")
        print(f"{launch_args=}")
    # Default values for CUDA backend (triton/backends/nvidia/compiler.py).
    num_warps = num_warps or 4
    num_stages = num_stages or 3
    triton_fn = _source_str_to_triton_function(source, func_name)
    sig, attrs, param_types = _derive_triton_signature_from_params(triton_fn, tir_params)
    compile_options = {"num_warps": num_warps, "num_stages": num_stages}
    fn_name, create_gpu_module, num_warps, shared_mem_size = _compile_triton_kernel(
        triton_fn, sig, {}, attrs, compile_options
    )
    launch_args, launch_param_tags = _derive_launch_args(launch_args, num_warps, shared_mem_size)
    kernel_module = create_gpu_module(param_types, launch_param_tags)
    # Convert launch_args to list[PrimExpr]. Python int automatically becomes runtime.BoxInt
    # when passed to C++, and we don't want that.
    launch_args = [convert(x) for x in launch_args]
    # Return a list.
    return [fn_name, kernel_module, launch_args]
