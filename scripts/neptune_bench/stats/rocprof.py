from pathlib import Path
from typing import Sequence

from perfetto.batch_trace_processor.api import BatchTraceProcessor
from perfetto.trace_processor import TraceProcessor

from .common import parse_dataframe_nvtx_context

SQL_QUERY = """
CREATE TEMP TABLE hip_api AS
  SELECT s.id, s.ts, s.dur, a.int_value AS corr
  FROM slice s
  JOIN args a ON a.arg_set_id = s.arg_set_id AND a.key = 'debug.corr_id'
  WHERE s.category = 'hip_api' AND s.name in ('hipLaunchKernel', 'hipModuleLaunchKernel');
CREATE INDEX hip_api_corr ON hip_api(corr);
CREATE INDEX hip_api_ts   ON hip_api(ts);

CREATE TEMP TABLE kernels AS
  SELECT s.id, s.ts, s.dur, s.name, a.int_value AS corr
  FROM slice s
  JOIN args a ON a.arg_set_id = s.arg_set_id AND a.key = 'debug.corr_id'
  WHERE s.category = 'kernel_dispatch';
CREATE INDEX kernels_corr ON kernels(corr);

CREATE TEMP TABLE roctx AS
  SELECT s.id, s.ts, s.dur, s.name
  FROM slice s WHERE s.category = 'marker_api';

CREATE TEMP TABLE hip_roctx_all AS
  SELECT h.corr, r.name AS roctx_name
  FROM hip_api h
  JOIN roctx r ON r.ts <= h.ts AND r.ts + r.dur >= h.ts;

SELECT
  k.ts AS kernel_start,
  k.dur / 1000.0 AS latency_us,
  k.name AS kernel_name,
  h.roctx_name AS ctx_marker
FROM kernels k JOIN hip_roctx_all h ON h.corr = k.corr
WHERE ctx_marker IS NOT NULL
ORDER BY kernel_start;"""


def pftrace_to_dataframe(profile: Path | str):
    with TraceProcessor(trace=str(profile)) as tp:
        df = tp.query(SQL_QUERY).as_pandas_dataframe()
    return parse_dataframe_nvtx_context(df)


def batch_pftrace_to_dataframes(profiles: Sequence[Path | str], chunk_size: int = 10):
    for i in range(0, len(profiles), chunk_size):
        chunk = [str(p) for p in profiles[i : i + chunk_size]]
        with BatchTraceProcessor(chunk) as btp:
            dfs = btp.query(SQL_QUERY)
            yield from [parse_dataframe_nvtx_context(df) for df in dfs]
