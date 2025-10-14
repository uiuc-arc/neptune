import logging
from abc import abstractmethod
from pathlib import Path
from typing import Callable

import torch
from tvm import ir, nd, neptune, tir
from tvm.meta_schedule.database import Database
from tvm.runtime import cuda
from tvm.runtime.ndarray import rocm

from ..runner import BenchmarkOperator, BenchmarkRunner
from ..utils import PrintToFileAfterPass, get_current_gpu_info

logger = logging.getLogger(__name__)


class NeptuneRunner(BenchmarkRunner):
    TUNING_DIR = Path("logs/neptune-tuning")
    DUMP_DIR = Path("logs/neptune-ir-dump")

    @classmethod
    def _create_from_schedulers(
        cls,
        operator: BenchmarkOperator,
        mod_creator: Callable[[tuple], ir.IRModule],
        schedulers: tuple,
    ):
        """Create 2N runners where N is the number of schedulers.
        Every 2 runners have the same schedule index, where the first runner uses the default schedule,
        and the second runner uses the tuned schedule (if any)."""
        runners = []
        for idx, scheduler in enumerate(schedulers):
            runners.append(cls(operator, mod_creator, idx, scheduler))
            runners.append(cls(operator, mod_creator, idx, None))
        return runners

    def __init__(
        self,
        operator: BenchmarkOperator,
        mod_creator: Callable[[tuple], ir.IRModule],
        schedule_idx: int,
        scheduler: Callable[[tir.Schedule], tir.Schedule] | None = None,
    ):
        mode = "tuned" if scheduler is None else "manual"
        impl_name = f"sch-{schedule_idx}-{mode}"
        super().__init__(operator, "neptune", language="neptune", impl_name=impl_name)
        self.mod_creator = mod_creator
        self.schedule_idx = schedule_idx
        self.scheduler = scheduler
        gpu_info = get_current_gpu_info()
        self.target = gpu_info.tvm_target
        self.device = rocm(0) if gpu_info.is_amd else cuda(0)
        self.executable: Callable | None = None
        self.inputs: list | None = None
        self.output_buf: nd.NDArray | None = None

    @abstractmethod
    def extract_shape(self, inputs: tuple) -> tuple: ...

    def _size_unsupported(self, shape: tuple) -> bool:
        return False

    def setup(self, *inputs) -> bool:
        import os

        shape = self.extract_shape(inputs)
        if self._size_unsupported(shape):
            return False
        mod = self.mod_creator(shape)
        if self.scheduler is None:
            sched = self._load_schedule_from_tuning(shape)
            if sched is None:
                return False
        else:
            sched = self._get_manual_schedule(mod)
        prefix = self._get_prefix(shape, self.DUMP_DIR)
        print_lower = os.environ.get("NEPTUNE_DUMP_LOWER_IR", "0") == "1"
        inst = [PrintToFileAfterPass(prefix)] if print_lower else []
        with ir.transform.PassContext(instruments=inst):
            self.executable = neptune.tvm_triton_build(sched.mod, self.target)
        self.inputs = [nd.array(x.cpu().numpy(), device=self.device) for x in inputs]
        self.output_buf = self._create_outbuf_from_mod(mod)
        return True

    def run(self):
        assert (
            self.executable is not None and self.output_buf is not None and self.inputs is not None
        )
        with self.mark_ctx():
            self.executable(*self.inputs, self.output_buf)

    def get_output(self) -> torch.Tensor:
        assert self.output_buf is not None
        return torch.from_numpy(self.output_buf.numpy()).cuda()

    def ms_autotune(self, inputs: tuple, n_trials: int, output_dir: Path | None = None):
        import numpy as np
        from tvm.meta_schedule.database import TuningRecord

        def get_latency_us(rec: TuningRecord):
            assert (latencies := rec.run_secs) is not None
            return np.mean([float(x) for x in latencies]) * 1e6

        assert self.scheduler is not None, (
            "Cannot autotune with a schedule-loading runner. "
            "Use its counterpart where self.scheduler is not None."
        )
        shape = self.extract_shape(inputs)
        output_dir = output_dir or self._get_prefix(shape, self.TUNING_DIR)
        if self._size_unsupported(shape):
            logger.info(
                f"Skipping autotuning for (operator={self.operator.name}, schedule={self.schedule_idx}, "
                f"shape={shape}) because the input is too large for this schedule."
            )
            return None
        our_mod = self.mod_creator(shape)
        if output_dir.exists():
            logger.info(f"Skipping autotuning at {output_dir} because it already exists.")
            return
        database, workload = neptune.ms_tune_with_scheduler(
            our_mod, self.scheduler, self.target, output_dir.as_posix(), n_trials
        )
        top_k = database.get_top_k(workload, 10)
        latencies = np.array([get_latency_us(rec) for rec in top_k])
        logger.info(f"Top {len(latencies)} schedules: {latencies} us")
        return database

    def _create_outbuf_from_mod(self, mod: ir.IRModule) -> nd.NDArray:
        from tvm._ffi.runtime_ctypes import DataType

        def list_of_int(xs):
            return [int(x) for x in xs]

        ((_, func),) = mod.functions_items()
        param_map = func.buffer_map
        output_buffer = param_map[func.params[-1]]
        output_type = DataType.to_torch_dtype(output_buffer.dtype, torch)
        return nd.array(
            torch.zeros(list_of_int(output_buffer.shape), dtype=output_type),
            device=self.device,
        )

    def _load_schedule_from_tuning(self, shape: tuple):
        prefix = self._get_prefix(shape, self.TUNING_DIR)
        if not (prefix / "database_tuning_record.json").exists():
            logger.info(f"No tuning records found at {prefix}; skipping.")
            return None
        database = Database.create("json", work_dir=prefix.as_posix())
        sch = neptune.get_best_from_single_target_db(database)
        logger.info(f"Loaded schedule from {prefix}")
        return sch

    def _get_manual_schedule(self, our_mod: ir.IRModule):
        assert self.scheduler is not None
        ((func0_gv, _),) = our_mod.functions_items()
        sch = tir.Schedule(our_mod)
        sch.work_on(func0_gv.name_hint)
        return self.scheduler(sch)

    def _get_prefix(self, shape: tuple, prefix: Path) -> Path:
        shape_str = ",".join(str(x) for x in shape)
        return prefix / f"{self.operator.name}-{shape_str}-s{self.schedule_idx}"
