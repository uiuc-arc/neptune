from dataclasses import dataclass
from functools import total_ordering
from typing import Any, Hashable, Sequence

import numpy as np
import pandas as pd


@total_ordering
@dataclass(frozen=True)
class Shape:
    shape: tuple[int, ...]

    @classmethod
    def from_str(cls, shape_str: str) -> "Shape":
        return cls(tuple(int(x) for x in shape_str.split(",")))

    def __lt__(self, other: "Shape") -> bool:
        return self.shape < other.shape

    def __repr__(self) -> str:
        return f"Shape({self.shape})"


@total_ordering
@dataclass(frozen=True)
class PerfSamples:
    raw_data: list[float]
    unit: str
    higher_better: bool

    @classmethod
    def join(cls, samples: list["PerfSamples"]) -> "PerfSamples":
        assert len(samples) > 0
        assert all(samples[0].unit == sample.unit for sample in samples)
        new_data = np.concatenate([sample.raw_data for sample in samples])
        return cls(
            raw_data=new_data.tolist(),
            unit=samples[0].unit,
            higher_better=samples[0].higher_better,
        )

    @property
    def std(self) -> float | None:
        return float(np.std(self.raw_data)) if len(self.raw_data) > 1 else None

    @property
    def mean(self) -> float:
        return float(np.mean(self.raw_data))

    def __len__(self) -> int:
        return len(self.raw_data)

    def __repr__(self) -> str:
        unit = f" ({self.unit})" if self.unit else ""
        return f"PerfSamples(mean={self.mean}{unit}, std={self.std}{unit})"

    def __str__(self) -> str:
        unit = f" ({self.unit})" if self.unit else ""
        return f"{self.mean} ± {self.std}{unit}"

    def __lt__(self, other: "PerfSamples") -> bool:
        if self.higher_better:
            return self.mean < other.mean
        else:
            return self.mean > other.mean


def combine_impl_multikernels(data: pd.DataFrame):
    """An implementation may launch multiple kernels per run.
    For each (implementation, repeatIdx) group, combine the data points across the kernels."""

    def check_distinct_kernels_and_sum(group: pd.DataFrame):
        output_fields = {"latency_us": group["latency_us"].sum()}
        if "registersPerThread" in group:
            output_fields["maxRegisters"] = group["registersPerThread"].max()
        if "staticSharedMemoryKiB" in group:
            output_fields["maxSharedMemKiB"] = max(
                group["staticSharedMemoryKiB"].max(), group["dynamicSharedMemoryKiB"].max()
            )
        return pd.Series(output_fields)

    return (
        data.groupby(["implementation", "repeatIdx"])
        .apply(check_distinct_kernels_and_sum, include_groups=False)
        .reset_index()
    )


def group_repeats(
    data: pd.DataFrame,
    drop_warmup: bool = True,
    group_by: Sequence[str] = ("implementation", "kernelName"),
    drop_columns: Sequence[str] = ("kernelStart",),
):
    """Group repeats of run from the same implementation. Latencies are combined into a PerfSamples object,
    while other columns are asserted to be the same for all repeats."""

    def check_equal_get_first(column, data: pd.Series):
        unique_values = data.unique()
        assert len(unique_values) > 0, f"No values for column {column}"
        assert len(unique_values) == 1, f"Multiple values for the same column {column}"
        return unique_values[0]

    def group_to_row(group: pd.DataFrame):
        drop_cols_set = set(list(drop_columns) + ["latency_us", "repeatIdx"])
        group = group.sort_values("repeatIdx")
        lat_samples = PerfSamples(group["latency_us"].tolist(), unit="us", higher_better=False)
        all_columns: dict[Hashable, Any] = {"latency_us": lat_samples}
        for key, column in group.items():
            if key in drop_cols_set:
                continue
            all_columns[key] = check_equal_get_first(key, column)
        return pd.Series(all_columns)

    if drop_warmup:
        data = data[data["repeatIdx"] != "warmup"]
    return data.groupby(list(group_by)).apply(group_to_row, include_groups=False).reset_index()


def select_best_and_relabel(data: pd.DataFrame, mask: pd.Series, relabel_to: str):
    if not mask.any():
        return data
    # Return the row with the best performance. We're comparing PerfSamples objects,
    # which depends on the custom __lt__ method.
    best_index = data.loc[mask, "latency_us"].idxmax()
    data.loc[best_index, "implementation"] = relabel_to
    # Select all other impls plus the row of our impls that have the best performance
    selected = ~mask
    selected[best_index] = True
    return data.loc[selected].reset_index(drop=True)


def parse_nvtx_context(context: str) -> tuple[str, str]:
    if "[" not in context:
        return context, ""
    context, count = context.split("[")
    assert count.endswith("]")
    count = count[:-1]
    return context, count


def parse_dataframe_nvtx_context(df: pd.DataFrame):
    def col_applier(row):
        context, count = parse_nvtx_context(row)
        return pd.Series({"implementation": context, "repeatIdx": count})

    marker_col = df["ctx_marker"]
    return df.drop(columns=["ctx_marker"]).join(marker_col.apply(col_applier))


def convert_to_us(duration: float, unit: str) -> float:
    if unit == "ns":
        return duration / 1000.0
    elif unit == "us":
        return duration
    elif unit == "ms":
        return duration * 1000.0
    else:
        raise ValueError(f"Unknown unit: {unit}")
