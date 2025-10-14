import logging
from pathlib import Path
from typing import Callable

from tvm import ir, target, tir
from tvm.meta_schedule.database import Database

logger = logging.getLogger(__name__)


def get_best_from_single_target_db(database: Database):
    from tvm.tir.schedule import Trace

    wkld = database.get_all_tuning_records()[0].workload
    top_k_records = database.get_top_k(wkld, 1)
    sch = tir.Schedule(wkld.mod)
    best_trace: Trace = top_k_records[0].trace
    best_trace.apply_to_schedule(sch, remove_postproc=False)
    return sch


def tvm_triton_build(mod, target: target.Target):
    from torch.utils._triton import has_triton

    from tvm.meta_schedule.builder.local_builder import default_build
    from tvm.tir import transform as tr

    assert has_triton(), "This function requires Triton to be installed and working."
    # Lower the module to prepare it for the TritonBuildKernel pass.
    mod = _tvm_triton_lower(mod, target)
    mod = tr.TritonBuildKernel()(mod)
    # Call default_build to transform untouched functions.
    return default_build(mod, target, None)


def extract_triton_kernels_as_strings(mod, target: target.Target):
    from tvm.tir import transform as tr

    # Lower the module to prepare it for the TritonCollectKernel function.
    mod = _tvm_triton_lower(mod, target)
    return tr.TritonCollectKernel(mod)


def extract_triton_kernels_as_torch_fns(mod, target: target.Target):
    import triton
    from torch.library import wrap_triton

    from tvm.runtime.triton import _source_str_to_triton_function
    from tvm.tir import transform as tr

    # Lower the module to prepare it for the TritonCollectKernel function.
    mod = _tvm_triton_lower(mod, target)
    torch_fns = []
    for name, (source, num_warps, num_stages, grid) in tr.TritonCollectKernel(mod).items():
        triton_callable = _source_str_to_triton_function(source, name)
        triton_callable = triton.autotune(
            configs=[triton.Config({}, num_warps=num_warps, num_stages=num_stages)], key=[]
        )(triton_callable)
        grid_int = tuple(int(x) for x in grid)
        torch_fn = wrap_triton(triton_callable)[grid_int]  # type: ignore
        torch_fns.append(torch_fn)
    # TODO: multiple kernels need to compose sometimes.
    return torch_fns


def _tvm_triton_lower(mod: ir.IRModule, target: target.Target):
    from tvm.ir.transform import Sequential
    from tvm.tir import transform as tr

    if target.host is None:
        logger.warning(f"Target {target} has no host, using LLVM as default")
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
    return Sequential(passes)(mod)


def ms_builder_initializer():
    from tvm._ffi import register_func

    reg_name = "meta_schedule.builder.triton_build"
    # Running multiple tuning sessions may lead to multiple registrations of the same function.
    register_func(
        reg_name, lambda mod, target, params: tvm_triton_build(mod, target), override=True
    )
    return reg_name


def ms_tune_with_scheduler(
    mod: ir.IRModule,
    scheduler: Callable[[tir.Schedule], tir.Schedule],
    target,
    work_dir: str,
    n_trials: int = 256,
):
    from tvm import meta_schedule as ms
    from tvm.meta_schedule import relax_integration as ms_relax
    from tvm.meta_schedule.runner.config import EvaluatorConfig

    logger = logging.getLogger("tvm.meta_schedule")
    logger.setLevel(logging.DEBUG)

    ((func0_gv, _),) = mod.functions_items()
    task_name = func0_gv.name_hint
    ctx = ms.TuneContext(
        mod,
        target=target,
        space_generator=ms.space_generator.ScheduleFn(scheduler, postprocs=[]),
        search_strategy="evolutionary",
        task_name=task_name,
        logger=ms_relax.get_loggers_from_work_dir(work_dir, [task_name])[0],
    )

    eval_config = EvaluatorConfig(enable_cache_flush=True, repeat=100, number=1, min_repeat_ms=0)
    runner = ms.Runner.create("local", evaluator_config=eval_config)
    database = ms.Database.create("json", work_dir=work_dir)
    task_scheduler = ms.TaskScheduler.create("gradient")
    f_build_name = ms_builder_initializer()
    cost_model = ms.CostModel.create("xgb")
    with ms.Profiler() as profiler:
        task_scheduler.tune(
            tasks=[ctx],
            task_weights=[1.0],
            max_trials_global=n_trials,
            max_trials_per_task=n_trials,
            num_trials_per_iter=16,
            builder=ms.builder.LocalBuilder(
                f_build=f_build_name, initializer=ms_builder_initializer, timeout_sec=10
            ),
            runner=runner,
            measure_callbacks=ms.MeasureCallback.create("default"),
            database=database,
            cost_model=cost_model,
        )
    cost_model.save(str(Path(work_dir) / "model.bin"))
    logger = logging.getLogger("tvm.meta_schedule")
    logger.info("Time spent in tuning: %s", profiler.table())
    workload = database.commit_workload(mod)
    return database, workload
