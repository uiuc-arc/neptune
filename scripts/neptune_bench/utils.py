import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from logging import getLogger
from pathlib import Path

import torch
from tvm.ir.instrument import pass_instrument
from tvm.ir.module import IRModule
from tvm.ir.transform import PassInfo
from tvm.target import Target

logger = getLogger(__name__)


def relaunch_with_profiler(
    output_prefix: Path,
    profiler: str | None,
    gpu_metrics: bool,
    relaunch_args: list[str],
    nsys_wait: bool = True,
):
    import sys
    import time
    from subprocess import check_call
    from tempfile import TemporaryDirectory

    # "-u" to unbuffer stdout, so that we can see the progress of the run.
    relaunch_args = [sys.executable, "-u"] + relaunch_args
    if profiler == "ncu":
        if get_current_gpu_info().is_amd:
            raise ValueError("ncu profiler does not support AMD GPUs")
        if gpu_metrics:
            raise ValueError("ncu profiler does not support gpu-metrics")
        if output_prefix.with_suffix(".ncu-rep").exists():
            logger.info(f"Skipping {output_prefix}.ncu-rep because it already exists")
            return
        logger.info(f"Running with ncu; saving profile to {output_prefix}.ncu-rep")
        ncu_args = ["ncu", "--nvtx", "--set", "full", "-o", output_prefix]
        ncu_args += ["--profile-from-start", "off"]  # Profile only after cudaProfilerStart().
        check_call(ncu_args + relaunch_args)
    elif profiler == "nsys":
        if get_current_gpu_info().is_amd:
            raise ValueError("ncu profiler does not support AMD GPUs")
        if output_prefix.with_suffix(".nsys-rep").exists():
            logger.info(f"Skipping {output_prefix}.nsys-rep because it already exists")
            return
        logger.info(f"Running with nsys; saving profile to {output_prefix}.nsys-rep")
        # Select cuda, nvtx, and osrt traces.
        nsys_args = ["nsys", "profile", "-o", output_prefix, "--trace=cuda,nvtx,osrt"]
        if gpu_metrics:
            nsys_args += ["--gpu-metrics-devices=cuda-visible"]
        if nsys_wait:
            # Let NSYS sync with the main process of the workload
            # and return with the same exit code as that process.
            # Does not work well with full-DNN inference workloads because PyTorch does... something.
            nsys_args += ["--wait=primary"]
        check_call(nsys_args + relaunch_args)
    elif profiler == "rocprof":
        if not get_current_gpu_info().is_amd:
            raise ValueError("rocprof profiler does not support NVIDIA GPUs")
        if gpu_metrics:
            raise ValueError("rocprof profiler does not support gpu-metrics")
        if output_prefix.with_suffix(".pftrace").exists():
            logger.info(f"Skipping {output_prefix}.pftrace because it already exists")
            return
        # rocprofv3 wants to write the profile result of every process to a separate file,
        # while we want a single file. It's the best if we tell it to write to a temp dir,
        # then concatenate the files and move it to where we want.
        with TemporaryDirectory() as tmpdir:
            roctx_args = ["rocprofv3", "--runtime-trace", "--kernel-trace", "--kernel-rename"]
            roctx_args += ["--output-format", "pftrace", "-d", tmpdir]
            check_call(roctx_args + ["--"] + relaunch_args)
            # Believe it or not, `rocprofv3` has an async component that we don't know when it's done.
            # So we have to wait for it to finish.
            time.sleep(10)
            # Concatenate the files.
            with open(output_prefix.with_suffix(".pftrace"), "wb") as fout:
                # Use rglob. rocprofv3 likes to create another subdir (with our hostname).
                for file in Path(tmpdir).rglob("**/*.pftrace"):
                    fout.write(file.read_bytes())
    else:
        raise ValueError(f"Unknown profiler: {profiler}")


@dataclass
class GPUInfo:
    is_amd: bool
    tvm_target: Target
    gpu_clock: int | None
    mem_clock: int | None


@cache
def get_current_gpu_info():
    """Get some info of the current GPU (device 0)."""

    GPU_INFO = {
        "NVIDIA RTX A5000": ("nvidia/rtx-a6000", 1695, 8000),
        "NVIDIA RTX 6000 Ada Generation": ("nvidia/rtx-a6000", 2505, 9500),
        "NVIDIA A100-SXM4-40GB": ("nvidia/nvidia-a100", 1410, None),
        "NVIDIA H100 PCIe": ("nvidia/nvidia-h100", 1755, None),
    }

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available")
    device_name = torch.cuda.get_device_name(0)
    if device_name.startswith("AMD"):
        return GPUInfo(True, Target("rocm", host="llvm"), None, None)
    target_name, gpu_clock, mem_clock = GPU_INFO[device_name]
    return GPUInfo(False, Target(target_name, host="llvm"), gpu_clock, mem_clock)


class GPUClockLockGuard:
    @classmethod
    def from_current_gpu(cls):
        info = get_current_gpu_info()
        assert info.gpu_clock is not None
        return cls(info.gpu_clock, info.mem_clock)

    def __init__(self, graphics_freq: int, memory_freq: int | None, relative_gpu_id: int = 0):
        import os

        if (visible_devices := os.environ.get("CUDA_VISIBLE_DEVICES")) is not None:
            process_gpu_ids = visible_devices.split(",")
            self.gpu_id = int(process_gpu_ids[relative_gpu_id])
        else:
            self.gpu_id = relative_gpu_id
        self.graphics_freq, self.memory_freq = graphics_freq, memory_freq

    def __enter__(self):
        cmds = [
            f"sudo nvidia-smi -i {self.gpu_id} -lgc {self.graphics_freq}",
            f"sudo nvidia-smi -i {self.gpu_id} -pm 1",
        ]
        if self.memory_freq is not None:
            cmds.append(f"sudo nvidia-smi -i {self.gpu_id} -lmc {self.memory_freq}")
        for cmd in cmds:
            try:
                subprocess.check_output(cmd.split())
            except subprocess.CalledProcessError:
                logger.warning(f"Failed to set GPU clock frequency: {cmd}")
        self.check_gpu_freq(tolerance=10)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        cmds = [f"sudo nvidia-smi -i {self.gpu_id} -rgc"]
        if self.memory_freq is not None:
            cmds.append(f"sudo nvidia-smi -i {self.gpu_id} -rmc")
        for cmd in cmds:
            try:
                subprocess.check_output(cmd.split())
            except subprocess.CalledProcessError:
                logger.warning(f"Failed to reset GPU clock frequency: {cmd}")

    def check_gpu_freq(self, tolerance: int):
        cmd = "nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd.split()).decode("utf-8").strip()
        output = output.split("\n")[self.gpu_id]
        graphics_freq, memory_freq = map(int, output.split(","))
        if abs(graphics_freq - self.graphics_freq) > tolerance:
            raise RuntimeError(
                f"GPU graphics frequency {graphics_freq} MHz is not close to expected {self.graphics_freq} MHz"
            )
        if self.memory_freq is not None and abs(memory_freq - self.memory_freq) > tolerance:
            raise RuntimeError(
                f"GPU memory frequency {memory_freq} MHz is not close to expected {self.memory_freq} MHz"
            )


@pass_instrument
class PrintToFileAfterPass:
    # TODO: some passes can call other passes, and this instrument class may produce confusing output.

    def __init__(self, work_dir: Path | str):
        import shutil

        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.last_mod_str = None
        self.pass_num = 0

    def run_before_pass(self, mod: IRModule, info: PassInfo):
        logger.info(f"Running pass: {info.name}")
        if self.last_mod_str is not None:
            return
        with open(self.work_dir / "_initial.py", "w") as f:
            f.write(str(mod))

    def run_after_pass(self, mod: IRModule, info: PassInfo):
        if str(mod) != self.last_mod_str:
            with open(self.work_dir / f"{self.pass_num:02d}-{info.name}.py", "w") as f:
                f.write(str(mod))
        # Some passes can modify the IRModule in place, so we need to get the string (essentially a deep copy)
        self.last_mod_str = str(mod)
        self.pass_num += 1


def flush_cuda_l2_cache(n_bytes: int = 200 * 1024**2):
    buf = torch.empty(n_bytes, dtype=torch.uint8, device="cuda")
    buf.zero_()
    torch.cuda.synchronize()


@contextmanager
def profiler_ctx():
    use_cuda_profiler = not get_current_gpu_info().is_amd
    if use_cuda_profiler:
        torch.cuda.cudart().cudaProfilerStart()  # type: ignore
        yield
        torch.cuda.cudart().cudaProfilerStop()  # type: ignore
    else:
        # Don't do anything for AMD. hipProfilerStart() has been deprecated. We'll use roctx instead.
        yield


@cache
def get_roctx():
    import ctypes

    lib = ctypes.cdll.LoadLibrary("/opt/rocm/lib/librocprofiler-sdk-roctx.so")
    # Signatures
    lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
    lib.roctxRangePushA.restype = ctypes.c_int
    lib.roctxRangePop.argtypes = []
    lib.roctxRangePop.restype = ctypes.c_int
    return lib


@contextmanager
def roctx_range(msg):
    lib = get_roctx()
    rid = lib.roctxRangePushA(msg.encode("utf-8"))
    if rid < 0:
        raise RuntimeError("roctxRangePushA failed")
    try:
        yield
    finally:
        rc = lib.roctxRangePop()
        if rc != 0:
            raise RuntimeError(f"roctxRangePop returned {rc}")


def bnsh_expand_for_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    q_heads = q.shape[1]
    b, kv_heads, s, d = k.shape
    groups = q_heads // kv_heads
    kv_shape = (b, q_heads, s, d)
    k = k.unsqueeze(2).expand(-1, -1, groups, -1, -1).contiguous().view(kv_shape)
    v = v.unsqueeze(2).expand(-1, -1, groups, -1, -1).contiguous().view(kv_shape)
    return q, k, v
