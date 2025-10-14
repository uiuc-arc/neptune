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
import json
import tempfile

from tvm import __version__ as tvm_version

from .module import Module, load_module


def _format_tvm_module_metadata(kernel_name, arg_types, launch_param_tags):
    """Format the TVM module metadata."""
    tvm_metadata = """{{
        "tvm_version": "{version}",
        "func_info": {{
            "{kernel_name}": {{
                "name": "",
                "arg_types": {arg_types},
                "launch_param_tags": {launch_param_tags}
            }}
        }}
    }}""".format_map(
        {
            "version": tvm_version,
            "kernel_name": kernel_name,
            "arg_types": json.dumps(arg_types),
            "launch_param_tags": json.dumps(launch_param_tags),
        }
    )
    return tvm_metadata


def create_cuda_module(
    ptx: str, kernel_arg_types: list[str], launch_param_tags: list[str], kernel_name: str
) -> Module:
    """
    Create a CUDA module from PTX and metadata.

    Parameters
    ----------
    ptx : str
        The PTX code of the kernel.

    kernel_arg_types : List[str]
        The types of the kernel arguments.

    launch_param_tags : List[str]
        The tags of the launch parameters.

    kernel_name : str
        The name of the kernel.

    Returns
    -------
    kernel_module : Module
        The CUDA module.
    """
    tvm_metadata = _format_tvm_module_metadata(kernel_name, kernel_arg_types, launch_param_tags)
    with tempfile.TemporaryDirectory() as temp_dir:
        ptx_path = f"{temp_dir}/{kernel_name}.ptx"
        with open(ptx_path, "w") as f:
            f.write(ptx)
        with open(f"{temp_dir}/{kernel_name}.tvm_meta.json", "w") as f:
            f.write(tvm_metadata)
        kernel_module = load_module(ptx_path)
    return kernel_module


def create_rocm_module(
    hsaco: str, kernel_arg_types: list[str], launch_param_tags: list[str], kernel_name: str
) -> Module:
    """
    Create a ROCm module from HSACO and metadata.

    Parameters
    ----------
    hsaco : str
        The HSACO code of the kernel.

    kernel_arg_types : List[str]
        The types of the kernel arguments.

    launch_param_tags : List[str]
        The tags of the launch parameters.

    kernel_name : str
        The name of the kernel.

    Returns
    -------
    kernel_module : Module
        The ROCm module.
    """
    tvm_metadata = _format_tvm_module_metadata(kernel_name, kernel_arg_types, launch_param_tags)
    with tempfile.TemporaryDirectory() as temp_dir:
        hsaco_path = f"{temp_dir}/{kernel_name}.hsaco"
        # Write HSACO as binary since it's compiled binary code
        with open(hsaco_path, "wb") as f:
            f.write(hsaco if isinstance(hsaco, bytes) else hsaco.encode())
        with open(f"{temp_dir}/{kernel_name}.tvm_meta.json", "w") as f:
            f.write(tvm_metadata)
        kernel_module = load_module(hsaco_path)
    return kernel_module
