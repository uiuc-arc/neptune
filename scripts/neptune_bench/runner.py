import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .operators import BenchmarkOperator
from .utils import flush_cuda_l2_cache, get_current_gpu_info


@dataclass
class BenchmarkRunner(ABC):
    operator: BenchmarkOperator
    provider_name: str
    language: str | None = None
    impl_name: str | None = None
    run_number: int | None = None

    @abstractmethod
    def setup(self, *inputs) -> bool:
        """Setup and warmup the operator. Cache the input. Return False if the operator is not supported."""
        pass

    @abstractmethod
    def run(self):
        """Run the operator. Do not perform transformations on the output."""
        pass

    @abstractmethod
    def get_output(self) -> torch.Tensor:
        """Get the output of the operator. Perform necessary transformations on the output."""
        pass

    def repeat(self, n_repeats: int):
        from time import sleep, time

        # Critical to sleep after each run to avoid GPU frequency throttling.
        # Throttling seems to happen regardless of how we lock the GPU clock.
        # We'll get a rough estimate of each run's duration and sleep a multiple of it.
        for _ in range(n_repeats):
            flush_cuda_l2_cache()
            self._increment_run()
            begin = time()
            self.run()
            torch.cuda.synchronize()
            duration = time() - begin
            sleep(duration * 3)

    def mark_ctx(self):
        name = f"{self.provider_name}({self.impl_name})" if self.impl_name else self.provider_name
        name += "[warmup]" if self.run_number is None else f"[{self.run_number}]"
        if get_current_gpu_info().is_amd:
            from .utils import roctx_range

            return roctx_range(name)
        else:
            import nvtx

            color = "blue" if self.run_number is None else "green"
            return nvtx.annotate(name, color=color)

    def _increment_run(self):
        if self.run_number is None:
            self.run_number = 0
        else:
            self.run_number += 1

    def __str__(self):
        kwargs = {
            "operator": self.operator.name,
            "provider": self.provider_name,
        }
        if self.impl_name:
            kwargs["impl"] = self.impl_name
        if self.language:
            kwargs["language"] = self.language
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"BenchmarkRunner({kwargs_str})"

    __repr__ = __str__

    def free_members_memory(self):
        for name in list(self.__dict__.keys()):
            if isinstance(self.__dict__[name], torch.Tensor):
                delattr(self, name)
        gc.collect()
        torch.cuda.empty_cache()


@dataclass
class DirectOutputRunner(BenchmarkRunner):
    output: torch.Tensor | None = None

    @abstractmethod
    def run_output(self) -> torch.Tensor:
        pass

    def run(self):
        self.output = self.run_output()

    def get_output(self) -> torch.Tensor:
        assert self.output is not None, "Output not set"
        return self.output
