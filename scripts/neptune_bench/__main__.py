import logging
from pathlib import Path
from typing import Sequence

import torch

from . import BenchmarkOperator, BenchmarkRunner, triton
from . import operators as ops
from . import ours as nep
from .flashinfer import FlashInferRunner
from .torch.flex import FlexAlibiRunner, FlexAttentionRunner
from .torch.sdpa import AttentionAlgo, TorchDecodeGQARunner, TorchSDPARunner
from .tridao import TriDaoRunner

logger = logging.getLogger(__name__)


def torch_softcap(softcap: float = 10.0):
    return lambda score, b, h, q_idx, kv_idx: torch.tanh(score / softcap) * softcap


def flex_windowed_mask_cond(left_window_size: int):
    return lambda b, h, q_idx, kv_idx: (q_idx >= kv_idx) & (q_idx - kv_idx < left_window_size)


RUNNERS: dict[BenchmarkOperator, Sequence[BenchmarkRunner]] = {
    ops.PF_GLOBAL: [
        *nep.NeptuneAttentionRunner.create_flex_from_schedulers(ops.PF_GLOBAL),
        FlexAttentionRunner(ops.PF_GLOBAL),
        TorchSDPARunner(ops.PF_GLOBAL, AttentionAlgo.FLASH, False),
        TorchSDPARunner(ops.PF_GLOBAL, AttentionAlgo.CUDNN, False),
        triton.OpenAITritonRunner(ops.PF_GLOBAL, False),
        triton.TriDaoTritonRunner(ops.PF_GLOBAL, False),
        TriDaoRunner(ops.PF_GLOBAL, causal=False),
        FlashInferRunner(ops.PF_GLOBAL, is_causal=False, is_decode=False),
        # TVMPrefillRunner(ops.PF_GLOBAL, False),
    ],
    ops.PF_CAUSAL: [
        *nep.NeptuneAttentionRunner.create_flex_from_schedulers(
            ops.PF_CAUSAL, mask_cond=nep.causal_mask_cond
        ),
        FlexAttentionRunner(ops.PF_CAUSAL, mask_expr=nep.causal_mask_cond),
        TorchSDPARunner(ops.PF_CAUSAL, AttentionAlgo.FLASH, True),
        TorchSDPARunner(ops.PF_CAUSAL, AttentionAlgo.CUDNN, True),
        triton.OpenAITritonRunner(ops.PF_CAUSAL, True),
        triton.TriDaoTritonRunner(ops.PF_CAUSAL, True),
        TriDaoRunner(ops.PF_CAUSAL, causal=True),
        FlashInferRunner(ops.PF_CAUSAL, is_causal=True, is_decode=False),
        # TVMPrefillRunner(ops.PF_CAUSAL, True),
    ],
    ops.PF_GQA: [
        *nep.NeptuneGQARunner.create_flex_from_schedulers(
            ops.PF_GQA,
            mask_cond=lambda b, h0, h1, q_idx, kv_idx: q_idx >= kv_idx,
        ),
        FlexAttentionRunner(ops.PF_GQA, mask_expr=nep.causal_mask_cond, enable_gqa=True),
        TorchSDPARunner(ops.PF_GQA, AttentionAlgo.FLASH, True, is_gqa=True),
        TorchSDPARunner(ops.PF_GQA, AttentionAlgo.CUDNN, True, is_gqa=True),
        triton.OpenAITritonRunner(ops.PF_GQA, True),
        triton.TriDaoTritonRunner(ops.PF_GQA, True),
        # The following 2 runners don't make a distinction between GQA and non-GQA.
        TriDaoRunner(ops.PF_GQA, causal=True),
        FlashInferRunner(ops.PF_GQA, is_causal=True, is_decode=False),
        # TVM runs GQA by simply repeating the key and value tensors.
        # TVMPrefillRunner(ops.PF_GQA, True),
    ],
    ops.PF_ALIBI: [
        *nep.NeptuneAttentionRunner.create_from_schedulers(
            ops.PF_ALIBI, nep.create_alibi_attention, has_mask=True
        ),
        FlexAlibiRunner(ops.PF_ALIBI, is_decode=False),
        # TriDao Triton could support Alibi, but runs out of SMEM.
        TriDaoRunner(ops.PF_ALIBI, has_alibi=True),
    ],
    # This "softcap" operator is causal + GQA + softcap.
    ops.PF_SOFTCAP: [
        *nep.NeptuneGQARunner.e(
            ops.PF_SOFTCAP,
            score_mod=nep.tir_softcap(softcap=ops.PF_SOFTCAP.softcap),
            mask_cond=lambda b, h0, h1, q_idx, kv_idx: q_idx >= kv_idx,
        ),
        FlexAttentionRunner(
            ops.PF_SOFTCAP,
            score_mod=torch_softcap(softcap=ops.PF_SOFTCAP.softcap),
            mask_expr=nep.causal_mask_cond,
            enable_gqa=True,
        ),
    ],
    # This "windowed" operator is window + GQA + softcap.
    ops.PF_WINDOWED: [
        *nep.NeptuneGQARunner.create_flex_from_schedulers(
            ops.PF_WINDOWED,
            score_mod=nep.tir_softcap(softcap=ops.PF_SOFTCAP.softcap),
            mask_cond=nep.tir_windowed_mask_cond(ops.PF_WINDOWED.window_size),
        ),
        FlexAttentionRunner(
            ops.PF_WINDOWED,
            score_mod=torch_softcap(softcap=ops.PF_SOFTCAP.softcap),
            mask_expr=flex_windowed_mask_cond(ops.PF_WINDOWED.window_size),
            enable_gqa=True,
        ),
        TriDaoRunner(
            ops.PF_WINDOWED,
            window_size=(ops.PF_WINDOWED.window_size - 1, 0),
            softcap=ops.PF_SOFTCAP.softcap,
        ),
    ],
    # Decoding attention starts.
    ops.DC_CAUSAL: [
        *nep.NeptuneDecodeRunner.create_flex_from_schedulers(ops.DC_CAUSAL),
        FlexAttentionRunner(ops.DC_CAUSAL),
        # NOTE 1: In PyTorch, neither the "FlashAttn" nor "Cudnn" backends support decoding. Decoding is
        # handled by the so-called "mem efficient" backend.
        # NOTE 2: the `is_causal` flag of PyTorch SDPA should be set to False for decoding. If it's True,
        # only 1 token from K/V is attended to (which does not produce a meaningful output).
        TorchSDPARunner(ops.DC_CAUSAL, AttentionAlgo.MEM_EFFICIENT, False, False),
        TriDaoRunner(ops.DC_CAUSAL, causal=True),
        FlashInferRunner(ops.DC_CAUSAL, is_causal=False, is_decode=True),
        # TVMDecodeRunner(ops.DC_CAUSAL),
        triton.XformerDecodeRunner(ops.DC_CAUSAL),
    ],
    ops.DC_GQA: [
        *nep.NeptuneGQADecodeRunner.create_flex_from_schedulers(ops.DC_GQA),
        FlexAttentionRunner(ops.DC_GQA, enable_gqa=True),
        TriDaoRunner(ops.DC_GQA, causal=True),
        FlashInferRunner(ops.DC_GQA, is_causal=False, is_decode=True),
        TorchDecodeGQARunner(ops.DC_GQA),
        # TVMDecodeRunner(ops.DC_GQA),
    ],
    ops.DC_ALIBI: [
        *nep.NeptuneDecodeRunner.create_from_schedulers(
            ops.DC_ALIBI, lambda shape: nep.create_alibi_attention(shape)
        ),
        FlexAlibiRunner(ops.DC_ALIBI, is_decode=True),
        TriDaoRunner(ops.DC_ALIBI, has_alibi=True),
    ],
    ops.DC_SOFTCAP: [
        *nep.NeptuneGQADecodeRunner.create_flex_from_schedulers(
            ops.DC_SOFTCAP, score_mod=nep.tir_softcap(softcap=ops.DC_SOFTCAP.softcap)
        ),
        FlexAttentionRunner(
            ops.DC_SOFTCAP,
            score_mod=torch_softcap(softcap=ops.DC_SOFTCAP.softcap),
            enable_gqa=True,
        ),
    ],
}
RUNNERS_BY_OP_NAME = {op.name: (op, runners) for op, runners in RUNNERS.items()}


def run_operator(operator: str, shape: list[int], n_repeats: int):
    op, runners = RUNNERS_BY_OP_NAME[operator]
    assert len(runners) >= 1, f"No runners for operator {operator}"
    # The complete shape (often 6D for attention) is a "shape signature". It isn't the shape of any actual input.
    shape_sig = op.complete_shape(tuple(shape))
    inputs = op.create_inputs(shape_sig)
    # Set up all the runners and compare results in a single pass first, before running the actual repeats.
    # This has multiple benefits:
    # 1. It avoids CUDAGraph complaints about the result being overwritten.
    # 2. It allows nsys to capture only the repeats, continuously, in a single report.
    ref_output = None
    for runner in runners:
        mem_reserved_mb = torch.cuda.memory_reserved() / 1024**2
        logger.info("starting %s", runner)
        if mem_reserved_mb > 1024:
            logger.info("memory usage: %.3fMB", mem_reserved_mb)
        # Set up the runner. If it's not supported, skip it.
        if not runner.setup(*inputs):
            logger.warning(f"{operator}: {runner} does not support this setup")
            runner.free_members_memory()
            continue
        try:
            runner.run()
        except Exception as e:
            logger.warning(f"{operator}: {runner} failed with exception: {e}")
            runner.free_members_memory()
            continue
        this_output = runner.get_output()
        if ref_output is None:
            ref_output = this_output
        else:
            shape_match = ref_output.shape == this_output.shape
            assert shape_match, f"{operator}: {runner} shape mismatches against {runners[0]}"
            if not torch.allclose(ref_output.cpu(), this_output.cpu(), atol=1e-2, rtol=0):
                logger.warning(f"{operator}: {runner} result mismatch against {runners[0]}")
        del this_output
        # Run the actual repeats.
        runner.repeat(n_repeats)
        runner.free_members_memory()


# No support for running multiple operators at once -- the nvtx markers won't be unique.
def profile_operator(
    operator: str, shape: list[int], profiler: str | None, repeat: int, gpu_metrics: bool
):
    from .utils import relaunch_with_profiler

    if profiler is None:
        run_operator(operator, shape, repeat)
        return
    if profiler == "ncu" and repeat > 1:
        raise ValueError("ncu profiler does not support repeat > 1")
    # Get the operator, and use it to get the shape signature.
    op, _ = RUNNERS_BY_OP_NAME[operator]
    shape_str = ",".join(str(x) for x in shape)
    shape_sig = op.complete_shape(tuple(shape))
    output_prefix = Path("logs/profiles/")
    output_prefix.mkdir(parents=True, exist_ok=True)
    output_prefix = output_prefix / f"{operator}-{','.join(str(x) for x in shape_sig)}"
    relaunch_args = ["-m", __spec__.name, "profile", operator, shape_str, "--repeat", str(repeat)]
    relaunch_with_profiler(output_prefix, profiler, gpu_metrics, relaunch_args)


def tune_operator(operator: str, shape: list[int], n_trials: int):
    from .ours.runner import NeptuneRunner

    op, runners = RUNNERS_BY_OP_NAME[operator]
    inputs = op.create_inputs(op.complete_shape(tuple(shape)))
    runners = [r for r in runners if isinstance(r, NeptuneRunner) and r.scheduler is not None]
    logger.info(
        f"Tuning {operator} with shape {shape}; found {len(runners)} Neptune schedulers to tune"
    )
    for runner in runners:
        runner.ms_autotune(inputs, n_trials)


def parse_args():
    import argparse

    def parse_shape(s: str) -> list[int]:
        return list(int(x) for x in s.split(","))

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    operators = {op.name for op in RUNNERS.keys()}
    profile = subparsers.add_parser("profile")
    profile.add_argument("operator", type=str, choices=operators)
    profile.add_argument("shape", type=parse_shape)
    profile.add_argument("--repeat", type=int, default=1)
    profile.add_argument("--profiler", type=str, choices=["ncu", "nsys", "rocprof"])
    profile.add_argument("--gpu-metrics", action="store_true")
    profile.set_defaults(
        func=lambda args: profile_operator(
            args.operator, args.shape, args.profiler, args.repeat, args.gpu_metrics
        )
    )

    tune = subparsers.add_parser("tune")
    tune.add_argument("operator", type=str, choices=operators)
    tune.add_argument("shape", type=parse_shape)
    tune.add_argument("--n-trials", type=int, default=128)
    tune.set_defaults(func=lambda args: tune_operator(args.operator, args.shape, args.n_trials))
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.func(args)
