# Neptune: Advanced ML Operator Fusion for Locality and Parallelism on GPUs

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2510.08726-b31b1b.svg)](https://arxiv.org/abs/2510.08726)

**Neptune** (PLDI 2026) is a tensor kernel compiler for advanced machine-learning operator fusion on GPUs.
It provides a few variants of **advanced operator fusion** that enable fusion between reductions,
that would stop classic fusion algorithms in today's tensor compilers.
For attention, Neptune fusion generates fused kernels equivalent **FlashAttention** or **FlashDecoding**,
depending on the variant of fusion algorithm used.

Neptune is built around two ideas:

1. **Advanced reduction fusion:** Fuse computations that conventional legality checks reject because of fusion-preventing dependencies.
2. **Schedule + tile optimization:** Keep high-level transformations in a schedule compiler and lower the result to tile-level GPU code generation.

## Quick Start

### Prerequisites

- A machine with a GPU that [Triton](https://github.com/triton-lang/triton) supports
- CUDA >= 12
- Python >= 3.10 (virtual environment recommended)
- CMake >= 3.18
- LLVM >= 15 (< 20 recommended)
- A C++17 compatible compiler

### Run from Prebuilt Docker Images

We have prepared a pre-build docker image hosted on DockerHub.
To run with Docker, make sure you have the docker CLI and runtime installed,
and that your GPU is supported by the Docker Nvidia runtime.

1. Fetch the docker image `evanzhao16/neptune-env:nvidia-latest` and launch it with Nvidia runtime:

   ```bash
   docker run --gpus all -it evanzhao16/neptune-env:nvidia-latest
   ```

   This command should bring you into a zsh terminal at path `/workspace`, inside the docker container.

2. Activate the pre-configured Python virtual environment at `/workspace/venv`:

   ```bash
   . /workspace/venv/bin/activate
   ```

3. Check that `/workspace/neptune`, which is the code repository of Neptune, exists.
   Neptune is written in C, C++, and Python.
   The C/C++ part of the project has been built,
   and the Python part has been installed into the virtual environment.

   You can verify everything is working simply via:

   ```bash
   python3 -c "import tvm.neptune as neptune"
   ```

### Building From Source

Neptune is built on a mirror of TVM, and the installation process is similar to TVM's.
Here is a self-contained installation guide; for more details, also see
[TVM's installation guide](https://tvm.apache.org/docs/install/from_source.html).

Cloning Neptune:

```bash
git clone https://github.com/uiuc-arc/neptune.git
cd neptune
git submodule update --init --recursive
```

Building the C++ part of Neptune using cmake:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_LLVM=ON -DUSE_CUDA=ON -B build -S ./
cmake --build build -j$(nproc)
```

Installing the Python package of Neptune (using pip's editable mode for development):

```bash
pip install -e ".[xgboost,highlight,dev]" --config-settings editable_mode=strict
```

or simply

```bash
pip install ".[xgboost,highlight,dev]"
```

To verify the installation, try importing the Python package:

```bash
python3 -c "import tvm.neptune as neptune"
```

## Usage And Benchmarks

Try this command:

```bash
python3 -m scripts.neptune_bench profile prefill_global 1,8192
```

This command runs `scripts/neptune_bench/__main__.py`.
It benchmarks global attention with a batch size of 1 and sequence length of 8192:
builds / fetches kernels from Neptune and a list of baselines and runs each of them.

Use `-h` to see the usage of the script.
Feel free to replace `prefill_global` with other attention operators like `prefill_causal`
and `decode_causal`, and try out different shapes.

The list of attention operators is defined in
[`operators.py`](scripts/neptune_bench/operators.py#L90),
and the list of baselines that support each operator is defined in
[`__main__.py`](scripts/neptune_bench/__main__.py#L26).

- Profile the kernels with Nvidia Nsight System if you have that
  [installed](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html).
  Make sure `nsys` is in your PATH.
  Then this command profiles all kernels of a setup and produces an `nsys` report
  that you can view with the nsys GUI tool:

  ```bash
  python3 -m scripts.neptune_bench profile prefill_global 1,8192 --profile --repeat 20
  ```

## Project Structure

Neptune is built on top of TVM, and the project structure is similar to TVM's:

```text
.
├── _paper/        # Paper appendix and supplemental material
├── apps/          # TVM application examples and extensions
├── cmake/         # CMake support files
├── conda/         # Conda packaging/environment files
├── configs/       # Build and runtime configuration files
├── docker/        # Docker-related files
├── docs/          # Documentation
├── include/       # Public C++ headers
├── python/        # Python package and TVM/Neptune Python bindings
├── src/           # C++ implementation
├── tests/         # Unit and integration tests
└── 3rdparty/      # Third-party dependencies/submodules
```

## Citation

```bibtex
@article{zhao2026neptune,
  title   = {Neptune: Advanced ML Operator Fusion for Locality and Parallelism on GPUs},
  author  = {Zhao, Yifan and Johnson, Egan and Chatarasi, Prasanth and Adve, Vikram S. and Misailovic, Sasa},
  journal = {Proceedings of the ACM on Programming Languages},
  volume  = {10},
  number  = {PLDI},
  article = {220},
  year    = {2026},
  doi     = {10.1145/3808298}
}
```

## Paper and Resources

- Paper: https://dl.acm.org/doi/10.1145/3808298 or https://arxiv.org/abs/2510.08726 (with appendix)
- Appendix: `_paper/neptune-appendix.pdf`

## Acknowledgements

Neptune builds on ideas and infrastructure from schedule-based tensor compilers and tile-based GPU programming systems.
In particular, the paper describes an implementation built on top of TVM-style tensor expressions/schedules and Triton-style tile optimization/code generation.
