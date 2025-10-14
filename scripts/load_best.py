from pathlib import Path

import numpy as np
from tvm import meta_schedule as ms
from tvm import tir


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="+")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def print_outputs(inst):
    return tuple(str(output) for output in inst.outputs)


args = parse_args()
for path in args.path:
    print(path)
    database = ms.Database.create("json", work_dir=path)
    all_records = database.get_all_tuning_records()
    wkld = all_records[0].workload
    rt_mods = []
    for i, rec in enumerate(database.get_top_k(wkld, args.top_k)):
        sch = tir.Schedule(wkld.mod, enable_check=False)
        rec.trace.apply_to_schedule(sch, remove_postproc=False)
        assert (latencies := rec.run_secs) is not None
        latency_us = np.mean([float(x) for x in latencies]) * 1e6
        assert (trace := sch.trace) is not None
        decisions = sorted(
            [(print_outputs(inst), values) for inst, values in trace.decisions.items()]
        )
        print(f"Rank {i}: {latency_us=:.2f} us, {decisions=}")
