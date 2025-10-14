import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from neptune_bench import operators as ops
from neptune_bench.stats import common, nsys, rocprof
from neptune_bench.stats.common import PerfSamples, Shape
from scipy import stats
from scipy.stats import gmean


@dataclass
class ArchInfo:
    ident: str
    is_nvidia: bool

    @property
    def profile_path(self) -> Path:
        return Path(f"logs/results_{self.ident}/profiles")

    @property
    def suffix(self) -> str:
        return "nsys-rep" if self.is_nvidia else "pftrace"


ARCHS = {
    "RTX 6000 Ada": ArchInfo("6000ada", True),
    "RTX A5000": ArchInfo("a5000", True),
    "A100": ArchInfo("a100", True),
    "MI300": ArchInfo("mi300", False),
}
POPULAR_OPS = {
    ops.PF_GLOBAL.name: ops.PF_GLOBAL.with_name("Global (PF)"),
    ops.PF_CAUSAL.name: ops.PF_CAUSAL.with_name("Causal (PF)"),
    ops.PF_GQA.name: ops.PF_GQA.with_name("GQA (PF)"),
    ops.DC_CAUSAL.name: ops.DC_CAUSAL.with_name("Causal (DC)"),
    ops.DC_GQA.name: ops.DC_GQA.with_name("GQA (DC)"),
}
OTHER_OPS = {
    ops.PF_ALIBI.name: ops.PF_ALIBI.with_name("ALiBi (PF)"),
    ops.DC_ALIBI.name: ops.DC_ALIBI.with_name("ALiBi (DC)"),
    ops.PF_SOFTCAP.name: ops.PF_SOFTCAP.with_name("SoftCap (PF)"),
    ops.DC_SOFTCAP.name: ops.DC_SOFTCAP.with_name("SoftCap (DC)"),
    ops.PF_WINDOWED.name: ops.PF_WINDOWED.with_name("Window (PF)"),
}
ALL_OPS = {op: renamed_op for op, renamed_op in {**POPULAR_OPS, **OTHER_OPS}.items()}
LIB_SUPPORTED_OPS = {op.name for op in ALL_OPS.values()} - {"SoftCap (DC)", "SoftCap (PF)"}

# Order is important here because we want to apply colors in a specific order.
IMPL_COMPILERS = {
    "neptune": "Neptune",
    "flex": "FlexAttn",
    "triton(openai)": "OpenAI Triton",
    "triton(tri-dao)": "Tri-Dao Triton",
    "triton(xformer)": "Xformer Triton",
    "tvm": "TVM",
}
IMPL_LIBRARIES = {
    "neptune": "Neptune",
    "torch(cutlass)": "PyTorch",
    "torch(cudnn)": "cuDNN",
    "tridao": "Tri-Dao Cutlass",
    "flashinfer": "FlashInfer",
}
ALL_IMPLS = {**IMPL_COMPILERS, **IMPL_LIBRARIES}

# Create a consistent color mapping for all implementations
# Get default matplotlib colors but move green to the front
ALL_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
green_idx = 2  # The default green is at index 2
ALL_COLORS = [ALL_COLORS[green_idx]] + ALL_COLORS[:green_idx] + ALL_COLORS[green_idx + 1 :]
IMPL_COMPILER_COLORS = {
    impl: ALL_COLORS[i % len(ALL_COLORS)] for i, impl in enumerate(IMPL_COMPILERS.values())
}
IMPL_LIBRARY_COLORS = {
    impl: ALL_COLORS[i % len(ALL_COLORS)] for i, impl in enumerate(IMPL_LIBRARIES.values())
}
# plt.rcParams["axes.prop_cycle"] = cycler("color", colors)

# Set pandas print limit to 10000 rows, 10 columns and 200 chars per row:
pd.set_option("display.max_rows", 10000)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 200)
# Set global font sizes for matplotlib
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 20,
        "errorbar.capsize": 2,
        "lines.linewidth": 1.5,
    }
)
ERBAR_KW = {"ecolor": "black", "capthick": 1, "elinewidth": 1}


def make_line_plots(
    output_dir: Path, all_profiles: pd.DataFrame, plot_name_prefix: str, colormap: dict[str, str]
):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Filter to only compiler-based implementations, and relabel them.
    all_profiles = all_profiles[all_profiles["implementation"].isin(colormap.keys())].copy()
    # Extract seq_len from shape.
    all_profiles["seq_len"] = all_profiles["shape"].apply(lambda shape: shape.shape[4])
    # Operators are sorted in the order of ALL_OPS. Architectures are ordered by ARCHS.
    first_op = True
    all_profiles = sort_ops_by_global_order(all_profiles)
    for op_name, op_profiles in all_profiles.groupby("op_name", sort=False):
        n_archs = len(ARCHS)
        height = 3.7 if first_op else 3
        fig, axes = plt.subplots(1, n_archs, squeeze=False, figsize=(6 * n_archs, height))
        axes = axes.flatten()
        for idx, arch_name in enumerate(ARCHS.keys()):
            df = op_profiles[op_profiles["arch"] == arch_name]
            rel_perfs = rel_perf_over_impl(df, "Neptune")
            is_first_arch = idx == 0
            ax: Axes = axes[idx]
            rel_perf_line_plot(ax, rel_perfs, first_op, colormap)
            if is_first_arch:
                ax.set_ylabel("Relative Performance", labelpad=5)
            if first_op:
                ax.set_title(arch_name, pad=10)
        if first_op:
            fig_top_legend(fig, axes[0])
            first_op = False
        plt.tight_layout()
        plt.savefig(output_dir / f"{plot_name_prefix}_{op_name}.pdf", bbox_inches="tight")
        plt.close(fig)


def print_op_stats(profiles: pd.DataFrame):
    def compute_neptune_speedup(setup_df: pd.DataFrame):
        neptune_row = setup_df[setup_df["implementation"] == "Neptune"]
        assert len(neptune_row) == 1, f"Expected 1 row, got {setup_df}"
        neptune_lat = neptune_row.iloc[0]["latency_us"]
        other_rows = setup_df[setup_df["implementation"] != "Neptune"]
        # Again -- PerfSamples objects have custom __lt__ methods, so max() is the lowest latency.
        others_best_lat = other_rows["latency_us"].max()
        return others_best_lat.mean / neptune_lat.mean

    # For each setup, compute Neptune's speedup over the best "other" implementation.
    groups = profiles.groupby(["arch", "op_disp_name", "shape"])
    speedups = groups.apply(compute_neptune_speedup, include_groups=False)
    # `speedups` is a Series. Count how many speedups are greater than 1.0.
    print(
        f"Out of {len(speedups)} setups, Neptune is better than all other baselines "
        f"{len(speedups[speedups > 1.0])} times."
    )
    # Also calculate a global geomean speedup.
    print(f"Global geomean speedup: {gmean(speedups.values)}")
    # Calculate the geometric mean over all the shapes for each (arch, op_disp_name) pair,
    # and print the results as a LaTeX table.
    speedups = (
        speedups.groupby(level=["arch", "op_disp_name"])
        .apply(lambda x: gmean(x.values))
        .to_frame("speedup")
        .reset_index()
        .pivot(index="op_disp_name", columns="arch", values="speedup")
    )
    speedups = sort_ops_by_global_order(speedups)
    print(speedups.to_latex(float_format="%.2f"))
    # Geomean again over all ops.
    print(f"Global geomean speedup: {speedups.apply(lambda x: gmean(x.values))}")


def plot_3a_single(output_dir: Path):
    gqa_paths = list(Path("logs/results_a5000/profiles-thruput").glob("prefill_gqa-*.nsys-rep"))
    df = pd.concat([load_profile(path, "RTX A5000", True) for path in gqa_paths])
    df["batch_size"] = df["shape"].apply(lambda shape: shape.shape[0])
    assert len(unique_ops := df["op_name"].unique()) == 1
    op = ALL_OPS[unique_ops[0]]

    def tflops_sec(shape: Shape, latency_us: PerfSamples):
        # Get the flops of each operator using the Operator class, then convert to TFLOPS/s.
        thruputs = op.flops(shape.shape) / np.array(latency_us.raw_data) / 1e6
        return thruputs.mean()

    df["tflops_sec"] = df.apply(lambda row: tflops_sec(row["shape"], row["latency_us"]), axis=1)  # type: ignore
    df = sort_impls_by_global_order(df.set_index(["implementation", "batch_size"]))

    # Make a plot with a broken y-axis.
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 4.5), sharex=True, gridspec_kw={"height_ratios": [5, 2]}
    )
    thruput_line_plot(ax1, df)
    ax1.set_ylim(bottom=65, top=90)
    ax2.set_ylim(bottom=0.0, top=10)
    ax2.set_xlabel("Batch Size", labelpad=5)
    ax2.set_xscale("log", base=2)
    # Add grid for better readability
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax2.grid(True, linestyle="--", alpha=0.7)
    fig.legend(bbox_to_anchor=(0.25, 0.4), loc="center", ncols=1, fontsize=15)
    fig.supylabel("TFLOPS/s")
    finish_broken_yaxis(ax1, ax2)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1, left=0.09)
    plt.savefig(output_dir / "3a.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_3a_full_appendix(output_dir: Path):
    all_profiles = [
        load_all_profiles_from(
            Path(f"logs/results_{arch.ident}/profiles-thruput"), arch_name, arch.is_nvidia
        )
        for arch_name, arch in list(ARCHS.items())
    ]
    all_profiles = pd.concat(all_profiles)
    all_profiles["batch_size"] = all_profiles["shape"].apply(lambda shape: shape.shape[0])

    first_op = True
    all_profiles = sort_ops_by_global_order(all_profiles)
    for op_name, op_profiles in all_profiles.groupby("op_name", sort=False):

        def tflops_sec(row):
            # Get the flops of each operator using the Operator class, then convert to TFLOPS/s.
            op = ALL_OPS[cast(str, op_name)]
            shape, latency_us = row["shape"], row["latency_us"]
            thruputs = op.flops(shape.shape) / np.array(latency_us.raw_data) / 1e6
            return thruputs.mean()

        op_profiles["tflops_sec"] = op_profiles.apply(tflops_sec, axis=1)
        n_archs = len(ARCHS)
        height = 3.8 if first_op else 3.5
        fig, axes = plt.subplots(1, n_archs, squeeze=False, figsize=(6 * n_archs, height))
        axes = axes.flatten()
        for idx, arch_name in enumerate(ARCHS.keys()):
            df = op_profiles[op_profiles["arch"] == arch_name].copy()
            df = sort_impls_by_global_order(df.set_index(["implementation", "batch_size"]))
            ax = axes[idx]
            thruput_line_plot(ax, df)
            ax.set_xlabel("Batch Size", labelpad=3)
            ax.set_xscale("log", base=2)
            if idx == 0:
                ax.set_ylabel("TFLOPS/s", labelpad=3)
            if first_op:
                ax.set_title(arch_name, pad=10)
        if first_op:
            fig_top_legend(fig, axes[0])
            first_op = False
        plt.tight_layout()
        plt.savefig(output_dir / f"3a_{op_name}.pdf", bbox_inches="tight")
        plt.close(fig)


def get_mirage_data():
    df = load_all_profiles_from(Path("logs/results_a100/profiles-mirage"), "A100", True)
    # Drop the mirage-full bit for now.
    df = df[df["implementation"] != "mirage_full"]
    # Get relative performance of Neptune over Mirage, and drop the Mirage numbers (because that's always 1.0).
    rel_perf_df = rel_perf_over_impl(df, "mirage", ["op_disp_name", "shape"])
    rel_perf_df = rel_perf_df.loc["neptune"]
    # Drop the "shape" index into the columns, and extract Q-seq-len and KV-seq-len from the shape.
    rel_perf_df = rel_perf_df.reset_index(level="shape")
    shapes = rel_perf_df["shape"]
    rel_perf_df["q_seq_len"] = shapes.apply(lambda shape: shape.shape[3])
    rel_perf_df["kv_seq_len"] = shapes.apply(lambda shape: shape.shape[4])
    # Separate out the global and causal cases.
    rel_perf_df = rel_perf_df[["rel_perf", "q_seq_len", "kv_seq_len"]]
    global_df, causal_df = rel_perf_df.loc["Global (PF)"], rel_perf_df.loc["Causal (PF)"]
    return global_df, causal_df


def plot_ablation_pair(plots: Path, bar_width: float = 0.2):
    def extract_latency(df: pd.DataFrame, impl: str):
        return cast(PerfSamples, df.loc[impl, "latency_us"]).mean

    def load_one_ablation(abl: Path):
        data = nsys.nsys_to_dataframe(abl)
        df = common.combine_impl_multikernels(data)
        df = common.group_repeats(df, group_by=["implementation"]).set_index("implementation")
        tvm = extract_latency(df, "tvm")
        no_transform = extract_latency(df, "neptune(sch-0-manual)")
        no_tuning = extract_latency(df, "neptune(sch-1-manual)")
        full = min(extract_latency(df, "neptune(sch-1-tuned)"), no_tuning)
        return [full / t for t in [tvm, no_transform, no_tuning, full]]

    xlabels = [
        "TVM",
        "Tile Optim.\nNo Fusion",
        "Fusion\nNo Tuning",
        "Full Neptune",
    ]
    xs = np.arange(len(xlabels))
    abl1_data = load_one_ablation(
        Path("logs/results_6000ada/profiles/prefill_global-1,32,32,512,512,128.nsys-rep")
    )
    abl2_data = load_one_ablation(
        Path("logs/results_6000ada/profiles/prefill_causal-1,32,32,512,512,128.nsys-rep")
    )
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.bar(xs - bar_width / 2, abl1_data, label="Global (PF)", width=bar_width)
    ax.bar(xs + bar_width / 2, abl2_data, label="Causal (PF)", width=bar_width)
    ax.set_ylabel("Relative Performance")
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.legend(bbox_to_anchor=(0.2, 0.75), loc="center", ncols=1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    plt.savefig(plots / "4a.pdf", bbox_inches="tight")
    plt.close(fig)


def rel_perf_line_plot(ax: Axes, rel_perfs: pd.DataFrame, xlabel: bool, colormap: dict[str, str]):
    # Plot a line for each implementation
    for impl, color in colormap.items():
        xs, ys, errs = [], [], []
        if impl in rel_perfs.index.get_level_values("implementation"):
            # Sort by seq_len.
            impl_data = rel_perfs.loc[impl]
            assert impl_data.index.name == "seq_len"
            impl_data = impl_data.sort_index()
            xs = impl_data.index.values
            ys = np.array(impl_data["rel_perf"])
            errs = np.array(impl_data["rel_perf_ci"])
        ax_errorbar(ax, xs, ys, errs, impl, color=color)
    if xlabel:
        ax.set_xlabel("Sequence Length", labelpad=5)
    ax.set_xscale("log", base=2)
    # Add grid for better readability
    ax.grid(True, linestyle="--", alpha=0.7)
    # Add a horizontal line at y=1.0 to show the baseline
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    # Set y-axis to start at 0.0, and if it stops at less than 1.05, set it to 1.05.
    ax.set_ylim(bottom=0.0, top=max(1.05, ax.get_ylim()[1]))


def thruput_line_plot(ax: Axes, thruputs: pd.DataFrame):
    # Plot a line for each implementation
    groups = thruputs.groupby(level="implementation", sort=False)
    for color, (impl, impl_data) in zip(ALL_COLORS, groups):
        impl_data = impl_data.loc[impl]
        # Sort by batch size.
        assert impl_data.index.name == "batch_size"
        impl_data = impl_data.sort_index()
        xs = impl_data.index.values
        ys = np.array(impl_data["tflops_sec"])
        ax_errorbar(ax, xs, ys, [0] * len(xs), impl, color=color)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", alpha=0.7)
    return len(groups)


def full_bar_plot(data: pd.DataFrame, ax: Axes, set_labels_legend: bool, bar_width: float = 0.1):
    # Calculate bar positions for each implementation
    op_centers = {}  # op_name -> center position for x-tick
    data = sort_ops_by_global_order(data)
    for i, (display_name, op_df) in enumerate(data.groupby("op_disp_name", sort=False)):
        # Assign relative positions to each implementation that exists for this variant
        op_df = sort_impls_by_global_order(op_df)
        assert len(op_df.index.unique()) == (N := len(op_df.index)), (
            "Expecting 1 row per (impl, op_disp_name)"
        )
        bar_pos = np.arange(N) * bar_width
        # Calculate the starting and center positions for this impl's bars
        op_centers[display_name] = i
        data.loc[op_df.index, "bar_pos"] = bar_pos + i - max(bar_pos) / 2
    # Plot the bars
    data = sort_impls_by_global_order(data)
    for impl, impl_df in data.groupby("implementation", sort=False):
        y_pos = np.array(impl_df["bar_pos"])
        means = np.array(impl_df["rel_perf_gm"])
        errors = np.array(impl_df["rel_perf_gm_ci"])
        color = IMPL_LIBRARY_COLORS[impl]  # type: ignore
        ax.barh(y_pos, means, bar_width, xerr=errors, label=impl, color=color)
    # Set y-ticks at the center of each variant's bars
    if set_labels_legend:
        ax.set_yticks(list(op_centers.values()))
        ax.set_yticklabels(list(op_centers.keys()), rotation=30, ha="right")
    else:
        ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlim(left=0.0)
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)


def rel_perf_over_impl(
    data: pd.DataFrame, relative_to: str, groupby_tail: Sequence[str] = ("seq_len",)
):
    # Create a Series with MultiIndex (impl, seq_len) and PerfSamples values.
    # Verify that each (impl, seq_len) combination has exactly one row.
    groupby = ["implementation", *groupby_tail]
    group_counts = data.groupby(groupby).size()
    if not (group_counts == 1).all():
        invalid_groups = group_counts[group_counts != 1]  # type: ignore
        raise AssertionError(
            f"Expected 1 row for each pair of (impl, seq_len), but found groups with counts: {invalid_groups.to_dict()}"
        )
    latencies = data.set_index(groupby)["latency_us"]

    def compute_stats_row(row_data):
        # Get the PerfSamples value from the single-column row
        this_lat: PerfSamples = row_data.iloc[0]
        impl, *rest = row_data.name
        # For the reference impl, relative performance is 1.0 with zero uncertainty.
        if impl == relative_to:
            return pd.Series({"rel_perf": 1.0, "rel_perf_ci": 0.0, "log_rel_perf_var": 0.0})
        # If the reference impl doesn't have any data, return NaN.
        base_key = (relative_to, *rest)
        if base_key not in latencies.index:
            return pd.Series(
                {"rel_perf": np.nan, "rel_perf_ci": np.nan, "log_rel_perf_var": np.nan}
            )
        base_lat: PerfSamples = latencies.loc[base_key]
        assert (M := len(base_lat.raw_data)) == len(this_lat.raw_data) > 0, (
            f"Different number of runs for {relative_to} and {impl} at {rest}: {M} vs {len(this_lat.raw_data)}"
        )
        # An estimate of the relative performance (our_lat / cur_lat) is just the ratio of means.
        rel_perf = (our_mean := base_lat.mean) / (cur_mean := this_lat.mean)
        # The "delta" method for estimating the variance of `rel_perf` (assuming no correlation)
        sxx = np.var(base_lat.raw_data) / M
        syy = np.var(this_lat.raw_data) / M
        rel_var = sxx / cur_mean**2 + our_mean**2 * syy / cur_mean**4
        # 95% confidence interval; use t-distribution with M-1 degrees of freedom.
        rel_ci = stats.t.ppf(0.975, M - 1) * np.sqrt(rel_var)
        # Using the delta method again to estimate the log of the relative performance.
        log_rel_var = sxx / our_mean**2 + syy / cur_mean**2
        return pd.Series(
            {"rel_perf": rel_perf, "rel_perf_ci": rel_ci, "log_rel_perf_var": log_rel_var}
        )

    # Estimate relative performance, with stats for the confidence interval.
    return latencies.to_frame().apply(compute_stats_row, axis=1)


def geomean_over_seqlens(
    rel_perfs: pd.DataFrame, relative_to: str, extra_groupby: Sequence[str] = ()
):
    groupby = ["implementation", *extra_groupby]

    def compute_stats_group(per_impl):
        group_name = per_impl.name
        impl = group_name[0] if len(groupby) > 1 else group_name
        if impl == relative_to:
            return pd.Series({"rel_perf_gm": 1.0, "rel_perf_gm_ci": 0.0})
        log_data = np.log(np.array(per_impl["rel_perf"]))
        log_mean = np.mean(log_data)
        geomean = np.exp(log_mean)
        geomean_ci = min(geomean - np.exp(np.min(log_data)), np.exp(np.max(log_data)) - geomean)
        return pd.Series({"rel_perf_gm": geomean, "rel_perf_gm_ci": geomean_ci})

    return rel_perfs.groupby(level=groupby).apply(compute_stats_group)


def load_profile(profile: Path | list[Path], arch: str, is_nvidia: bool):
    def process_single_profile(profile: Path, data: pd.DataFrame):
        if len(data) == 0:
            warnings.warn(f"No data found in profile {profile}")
            return data
        combined_kernels = common.combine_impl_multikernels(data)
        combined_repeats = common.group_repeats(combined_kernels, group_by=["implementation"])
        # Pick the best Neptune schedule and relabel it to "neptune".
        data = common.select_best_and_relabel(
            combined_repeats,
            combined_repeats["implementation"].str.startswith("neptune("),
            relabel_to="neptune",
        )
        # Pick the best between torch(flash) and torch(mem-eff), and relabel it to torch(cutlass).
        data = common.select_best_and_relabel(
            data,
            data["implementation"].isin(["torch(flash)", "torch(mem-eff)"]),
            relabel_to="torch(cutlass)",
        )
        # Add metadata: architecture, operator, shape.
        data["arch"] = arch
        op_name, shape_str, *_ = profile.stem.split("-")
        data["op_name"] = op_name
        data["op_disp_name"] = ALL_OPS[op_name].name
        data["shape"] = Shape.from_str(shape_str)
        data["implementation"] = data["implementation"].map(ALL_IMPLS)
        return data

    if isinstance(profile, list):
        if is_nvidia:
            return pd.concat(
                [process_single_profile(p, nsys.nsys_to_dataframe(p)) for p in profile]
            )
        else:
            dfs = rocprof.batch_pftrace_to_dataframes(profile)
            return pd.concat([process_single_profile(p, df) for p, df in zip(profile, dfs)])
    else:
        data = (
            nsys.nsys_to_dataframe(profile) if is_nvidia else rocprof.pftrace_to_dataframe(profile)
        )
        return process_single_profile(profile, data)


def load_all_profiles_from(path: Path, arch: str, is_nvidia: bool):
    suffix = "nsys-rep" if is_nvidia else "pftrace"
    profiles = list(path.glob(f"*.{suffix}"))
    print(f"Loading {len(profiles)} profiles from {path}...")
    return load_profile(profiles, arch, is_nvidia)


def load_all_arch_profiles():
    print(f"Loading profiles ({len(ARCHS)} archs)...")
    ret = pd.concat(
        [
            load_all_profiles_from(arch_info.profile_path, disp_name, arch_info.is_nvidia)
            for disp_name, arch_info in ARCHS.items()
        ]
    )
    print("Loaded.")
    return ret


def sort_ops_by_global_order(df: pd.DataFrame):
    all_ops_order = {op.name: idx for idx, op in enumerate(ALL_OPS.values())}
    if "op_disp_name" not in df.index.names:
        df = df.set_index("op_disp_name")
    return df.sort_index(level="op_disp_name", key=lambda op_name: op_name.map(all_ops_order))


def sort_impls_by_global_order(df: pd.DataFrame):
    all_impls_order = {impl: idx for idx, impl in enumerate(ALL_IMPLS.values())}
    return df.sort_index(level="implementation", key=lambda impl: impl.map(all_impls_order))


def ax_errorbar(ax: Axes, x, y, yerr, label, **kwargs):
    ax.errorbar(
        x,
        y,
        yerr,
        marker="o",
        label=label,
        linewidth=1.5,
        ecolor="black",
        elinewidth=1,
        capthick=1,
        **kwargs,
    )


def row_major_legend(ax: Axes, n_rows: int, **legend_kwargs):
    # Get legend handles and labels
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    n_items = len(handles)
    n_cols = int(np.ceil(n_items / n_rows))
    # Rearrange for row-major order
    reordered_handles = []
    reordered_labels = []
    for col in range(n_cols):
        for row in range(n_rows):
            idx = row * n_cols + col
            if idx < n_items:
                reordered_handles.append(handles[idx])
                reordered_labels.append(labels[idx])
    ax.legend(reordered_handles, reordered_labels, ncols=n_cols, **legend_kwargs)


def fig_top_legend(fig: Figure, ax0: Axes, y_anchor: float = 1.03, **legend_kwargs):
    ax0.legend()
    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, y_anchor),
        loc="center",
        ncols=len(labels),
        **legend_kwargs,
    )
    ax0.legend().remove()


def pad_2d_dataframe(df_2d: pd.DataFrame, q_seq_len_range=None, kv_seq_len_range=None):
    """
    Pad a 2D DataFrame to have specified ranges for index and columns, filling with NaN.

    Args:
        df_2d: 2D DataFrame with q_seq_len as index and kv_seq_len as columns
        q_seq_len_range: Iterable of q_seq_len values (e.g., range(128, 8193, 128))
        kv_seq_len_range: Iterable of kv_seq_len values (e.g., range(128, 8193, 128))

    Returns:
        Padded DataFrame with NaN for missing combinations
    """
    if q_seq_len_range is not None:
        df_2d = df_2d.reindex(index=q_seq_len_range)
    if kv_seq_len_range is not None:
        df_2d = df_2d.reindex(columns=kv_seq_len_range)
    return df_2d


def finish_broken_yaxis(ax1: Axes, ax2: Axes):
    # hide the spines between ax1 and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    # fmt: off
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    # fmt: on
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)


def count_registers_smem(df: pd.DataFrame):
    neptune_rows = df[df["implementation"] == "Neptune"]
    assert len(neptune_rows) == 1
    neptune_row = neptune_rows.iloc[0]
    neptune_reg, neptune_smem = neptune_row["maxRegisters"], neptune_row["maxSharedMemKiB"]
    if neptune_smem == 0:
        neptune_smem = np.nan
    min_reg, max_reg = df["maxRegisters"].min(), df["maxRegisters"].max()
    min_smem, max_smem = df["maxSharedMemKiB"].min(), df["maxSharedMemKiB"].max()
    return pd.Series(
        {
            "reg_over_min": neptune_reg / min_reg,
            "reg_over_max": neptune_reg / max_reg,
            "smem_over_min": neptune_smem / min_smem,
            "smem_over_max": neptune_smem / max_smem,
        }
    )


if __name__ == "__main__":
    plots = Path("logs/plots")
    all_profiles = load_all_arch_profiles()

    pop_op_profiles = all_profiles[all_profiles["op_name"].isin(POPULAR_OPS.keys())]
    make_line_plots(plots, pop_op_profiles, "1a", colormap=IMPL_COMPILER_COLORS)
    other_op_profiles = all_profiles[all_profiles["op_name"].isin(OTHER_OPS.keys())]
    make_line_plots(plots, other_op_profiles, "1b", colormap=IMPL_COMPILER_COLORS)
    make_line_plots(plots, all_profiles, "1c", colormap=IMPL_LIBRARY_COLORS)
    plot_3a_single(plots)
    plot_3a_full_appendix(plots)
    plot_ablation_pair(plots)

    # get_mirage_data()
    # df = all_profiles.groupby(["arch", "op_disp_name", "shape"]).apply(
    #     count_registers_smem, include_groups=False
    # )
    # df = df.apply(lambda x: gmean(x.values[np.isfinite(x.values)]))
    # print(f"Register and SMEM usage: {df}")
    # print("Printing global stats for compiler baselines:")
    # print_op_stats(all_profiles[all_profiles["implementation"].isin(IMPL_COMPILERS.values())])
    # print("Printing global stats for library baselines:")
    # print_op_stats(
    #     all_profiles[
    #         all_profiles["implementation"].isin(IMPL_LIBRARIES.values())
    #         & all_profiles["op_disp_name"].isin(LIB_SUPPORTED_OPS)
    #     ]
    # )
