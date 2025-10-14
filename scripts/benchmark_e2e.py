import logging
import math
import sys
from pathlib import Path

import nvtx
import torch
import transformers as tfs
import tvm
from neptune_bench.ours import attn_gqa, attn_plain
from neptune_bench.utils import relaunch_with_profiler
from transformers.models.vit.modeling_vit import ViTAttention
from tvm.neptune import extract_triton_kernels_as_torch_fns

logger = logging.getLogger(__name__)
TARGET = tvm.target.Target("cuda", host="llvm")


class NeptuneDecoding(torch.nn.Module):
    def __init__(self, mod, target):
        super().__init__()
        self.fn1, self.fn2 = extract_triton_kernels_as_torch_fns(mod, target)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.dtype == k.dtype == v.dtype == torch.float16
        output = torch.empty_like(q)
        # TODO: hacky allocation sizes. We should get intermediate buffer sizes from the kernel.
        # In normal code paths, these allocations are handled by TVM runtime.
        rf_shape = (64, k.shape[1], 2)
        maxelem_rf = torch.empty(rf_shape, dtype=k.dtype, device=k.device)
        expsum_rf = torch.empty(rf_shape, dtype=k.dtype, device=k.device)
        batch_matmul_rf = torch.empty(*rf_shape, k.shape[3], dtype=k.dtype, device=k.device)
        self.fn1(q, k, v, batch_matmul_rf, expsum_rf, maxelem_rf)
        self.fn2(output, batch_matmul_rf, expsum_rf, maxelem_rf)
        return output


class NeptunePrefill(torch.nn.Module):
    def __init__(self, mod, target):
        super().__init__()
        (self.fn,) = extract_triton_kernels_as_torch_fns(mod, target)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.dtype == k.dtype == v.dtype == torch.float16
        output = torch.empty_like(q)
        self.fn(q, k, v, output)
        return output


def make_neptune_kernel(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    causal: bool,
    softcap: float | None,
    sliding_window: int | None,
):
    from neptune_bench.ours import tir_softcap, tir_windowed_mask_cond

    def _apply_scheduler(module: tvm.IRModule, scheduler):
        ((func0_gv, _),) = module.functions_items()
        sch = tvm.tir.Schedule(module)
        sch.work_on(func0_gv.name_hint)
        scheduled = scheduler(sch)
        return scheduled.mod

    bq, hq, sq, dq = q_shape
    bk, hk, sk, dk = k_shape
    assert bq == bk and dq == dk
    decode = sq == 1
    gqa = hq != hk

    score_mod = None if softcap is None else tir_softcap(softcap=softcap)
    if sliding_window is not None:
        mask_cond = tir_windowed_mask_cond(sliding_window)
    elif causal and not decode:
        mask_cond = attn_plain.causal_mask_cond
    else:
        mask_cond = None
    if gqa and decode:
        shape = (bq, hq, hk, sk, dq)
        module = attn_gqa.create_gqa_decoding(shape, score_mod, mask_cond)
        module = _apply_scheduler(module, attn_gqa.schedule_gqa_decoding)
    elif gqa and not decode:
        shape = (bq, hq, hk, sq, sk, dq)
        module = attn_gqa.create_grouped_query_attention(shape, score_mod, mask_cond)
        assert causal
        module = _apply_scheduler(module, attn_gqa.schedule_gqa)
    else:
        shape = (bq, hq, sq, sk, dq)
        module = attn_plain.create_general_attention(shape, score_mod, mask_cond)
        if decode:
            scheduler = attn_plain.schedule_flash_decoding
        elif causal:
            scheduler = attn_plain.schedule_mask_attn_flash
        else:
            scheduler = attn_plain.schedule_full_attn_flash
        module = _apply_scheduler(module, scheduler)
    if decode:
        return NeptuneDecoding(module, TARGET)
    else:
        return NeptunePrefill(module, TARGET)


NEPTUNE_MHA_CACHE = {}


def felix_attn_interface(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    sliding_window: int | None = None,
    softcap: float | None = None,
    **kwargs,
):
    from torch.nn.functional import pad

    causal = not isinstance(module, ViTAttention)
    assert key.shape == value.shape
    assert query.dtype == key.dtype == value.dtype == torch.float16
    if sliding_window is not None and sliding_window >= query.shape[2]:
        sliding_window = None  # Sliding window less than sequence length is unnecessary

    # TODO: our kernel should support padding
    need_pad = (seqlen := key.shape[2]) % 128 != 0
    if need_pad:
        # Only apply padding in prefill mode
        assert query.shape[2] == key.shape[2] == value.shape[2]
        next_size = math.ceil(seqlen / 128) * 128
        padding = (0, 0, next_size - seqlen, 0)
        query, key, value = pad(query, padding), pad(key, padding), pad(value, padding)

    cache_key = (*query.shape, *key.shape, softcap, sliding_window)
    if cache_key not in NEPTUNE_MHA_CACHE:
        kernel = make_neptune_kernel(query.shape, key.shape, causal, softcap, sliding_window)
        NEPTUNE_MHA_CACHE[cache_key] = kernel = kernel.to(key.device, dtype=key.dtype)
    else:
        kernel = NEPTUNE_MHA_CACHE[cache_key]
    ret = kernel(query, key, value)
    if need_pad:
        ret = ret[:, :, :seqlen, :]
    # PyTorch provides BNSH inputs but expect BSNH outputs, somehow. We'll transpose the output here.
    ret = ret.transpose(1, 2)
    return ret, None


def build_mpt_alibi_slopes(num_heads: int, alibi_bias_max: int = 8, device=None):
    r"""
    Link to paper: https://huggingface.co/papers/2108.12409 - Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation. This implementation has been copied from
    the alibi implementation of MPT source code that led to slightly different results than the Bloom alibi:
    https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L292
    """
    num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))
    base = torch.arange(1, num_heads_power_of_2 + 1, dtype=torch.int64, device=device).float()
    base = base * (alibi_bias_max / num_heads_power_of_2)

    slopes = 1.0 / torch.pow(2, base)
    slopes = slopes.view(1, num_heads_power_of_2, 1, 1)

    if num_heads_power_of_2 != num_heads:
        slopes = torch.concat([slopes[:, 1::2, ...], slopes[:, ::2, ...]], dim=1)
        slopes = slopes[:, :num_heads, ...]

    slopes = slopes.squeeze().contiguous()
    return slopes


def parse_args():
    import argparse

    ATTN_TYPES = set(["felix_attn", "flash_attention_2", "flex_attention", "sdpa"])
    parser = argparse.ArgumentParser(
        description="Run and profile an LLM with different types of attention."
    )
    parser.add_argument("model", type=str, choices=["llama", "gemma2", "vit", "mpt"])
    parser.add_argument("batch", type=int, help="Batch size for the model.")
    parser.add_argument("seqlen", type=int, help="Sequence length for the model.")

    def parse_attn_types(s: str):
        types = s.split(",")
        for ty in types:
            if ty not in ATTN_TYPES:
                parser.error(f"Invalid attention type: {ty}")
        return types

    parser.add_argument(
        "--attn-types",
        type=parse_attn_types,
        default="felix_attn,flash_attention_2,flex_attention,sdpa",
        help="Attention types to profile: felix_attn, flash_attention_2, flex_attention, sdpa. Defaults to everything.",
    )
    parser.add_argument(
        "--decode", action="store_true", help="Profile decoding if True, prefill if False."
    )
    parser.add_argument(
        "--repeat", type=int, default=10, help="Number of times to repeat the run for profiling."
    )
    parser.add_argument("--profiler", type=str, choices=["nsys", "rocprof"])
    parser.add_argument(
        "--gpu-metrics", action="store_true", help="Enable gpu metrics for profiling."
    )
    args = parser.parse_args()
    if args.model == "vit" and args.decode:
        parser.error("Decoding is not supported for ViT.")
    return args


def get_inputs(config, model, batch_size: int, seqlen: int, decode: bool):
    from copy import deepcopy

    f16 = torch.float16
    if decode:
        embed_shape = batch_size, seqlen - 1, config.hidden_size
        input_embeds = torch.randn(embed_shape, dtype=f16, device="cuda")
        # For decode inputs, run the model once on the prefill inputs (with 1 fewer token)
        # so we can get the KV cache.
        past_kv = model(inputs_embeds=input_embeds).past_key_values
        assert isinstance(past_kv, tfs.DynamicCache)
        dummy_next_token = torch.randn((batch_size, 1, config.hidden_size), dtype=f16).cuda()
        # Return a function that returns the right inputs for the model. Make a deepcopy of the KV cache
        # on each call because the model will modify it in-place.
        return lambda: {"inputs_embeds": dummy_next_token, "past_key_values": deepcopy(past_kv)}
    else:
        embed_shape = batch_size, seqlen, config.hidden_size
        input_embeds = torch.randn(embed_shape, dtype=f16, device="cuda")
        return lambda: {"inputs_embeds": input_embeds}


def get_vit_image_inputs(config, batch_size: int):
    return lambda: {
        "pixel_values": torch.randn(
            (batch_size, config.num_channels, config.image_size, config.image_size),
            dtype=torch.float16,
            device="cuda",
        )
    }


@torch.inference_mode()
def main():
    import torch._dynamo.config
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    torch._dynamo.config.cache_size_limit = 300
    ALL_ATTENTION_FUNCTIONS.register("felix_attn", felix_attn_interface)
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    model = args.model
    batch_size, seqlen = args.batch, args.seqlen
    attn_types, decode, repeat = args.attn_types, args.decode, args.repeat

    if args.profiler:
        output_prefix = Path("logs/e2e_profiles/")
        output_prefix.mkdir(parents=True, exist_ok=True)
        mode = "decode" if decode else "prefill"
        output_prefix = output_prefix / f"{model}-{mode}-{batch_size},{seqlen}"
        script_path = Path(sys.argv[0]).resolve()
        relaunch_args = [script_path, model, str(batch_size), str(seqlen), "--repeat", str(repeat)]
        relaunch_args += ["--attn-types", ",".join(attn_types)]
        if decode:
            relaunch_args.append("--decode")
        relaunch_with_profiler(
            output_prefix, args.profiler, args.gpu_metrics, relaunch_args, nsys_wait=False
        )
        return

    logger.info(f"{batch_size=}, {seqlen=}, {attn_types=}, {decode=}, {repeat=}")
    if model == "llama":
        # Llama 7B
        config = tfs.LlamaConfig(torch_dtype=torch.float16)
        logger.info("Creating the model...")
        model = tfs.LlamaModel(config).eval().to(dtype=torch.float16, device="cuda")  # type: ignore
        input_creator = get_inputs(config, model, batch_size, seqlen, decode)
    elif model == "gemma2":
        # Gemma2 27B
        config = tfs.Gemma2Config(
            num_hidden_layers=46,
            num_attention_heads=32,
            num_key_value_heads=16,
            head_dim=128,
            sliding_window=4096,
            hidden_size=4608,
            torch_dtype=torch.float16,
        )
        logger.info("Creating the model...")
        model = tfs.Gemma2Model(config).eval().to(dtype=torch.float16, device="cuda")  # type: ignore
        input_creator = get_inputs(config, model, batch_size, seqlen, decode)
    elif model == "vit":
        assert not decode
        img_size = math.floor(math.sqrt(seqlen - 1) * 16)
        logger.info(f"For ViT/L: equivalent image size = {img_size}")
        config = tfs.ViTConfig(
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            hidden_size=1024,
            image_size=img_size,
        )
        logger.info("Creating the model...")
        model = tfs.ViTModel(config).eval().to(dtype=torch.float16, device="cuda")  # type: ignore
        input_creator = get_vit_image_inputs(config, batch_size)
    elif model == "mpt":
        config = tfs.MptConfig(n_layers=1, torch_dtype=torch.float16, max_seq_len=seqlen)
        logger.info("Creating the model...")
        model = tfs.MptModel(config).eval().to(dtype=torch.float16, device="cuda")  # type: ignore
        input_creator = get_inputs(config, model, batch_size, seqlen, decode)

    for attn_type in attn_types:
        # Set the attention type, compile the model, and run again to make the actual compilation (JIT) happen
        logger.info(f"Running {attn_type=}: compiling the model...")
        model.config._attn_implementation = attn_type
        compiled_model = torch.compile(model, dynamic=False, fullgraph=False)
        compiled_model(**input_creator())
        # Start the actual runs
        logger.info("Starting timed runs...")
        for i in range(repeat):
            with nvtx.annotate(f"{attn_type=}[{i}]"):
                compiled_model(**input_creator())
        logger.info(f"{attn_type=} finished")


if __name__ == "__main__":
    main()
