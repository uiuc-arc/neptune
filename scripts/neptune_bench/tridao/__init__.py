import flash_attn
import torch

from ..runner import BenchmarkOperator, BenchmarkRunner


def _transform_io(t: torch.Tensor):
    return torch.permute(t, (0, 2, 1, 3))


class TriDaoRunner(BenchmarkRunner):
    def __init__(self, operator: BenchmarkOperator, has_alibi: bool = False, **runner_kwargs):
        super().__init__(operator, "tridao", "cutlass")
        self.runner_kwargs = runner_kwargs
        self.has_alibi = has_alibi

    def setup(self, *inputs) -> bool:
        if self.has_alibi:
            q, k, v, alibi_slopes = inputs
            self.runner_kwargs["alibi_slopes"] = alibi_slopes
        else:
            q, k, v = inputs
        self.q, self.k, self.v = map(_transform_io, (q, k, v))
        return True

    def run(self):
        with self.mark_ctx():
            self.output = flash_attn.flash_attn_func(self.q, self.k, self.v, **self.runner_kwargs)

    def get_output(self) -> torch.Tensor:
        assert self.output is not None, "Output not set"
        return _transform_io(self.output)  # type: ignore
