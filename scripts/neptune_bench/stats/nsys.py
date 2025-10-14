import sqlite3
from pathlib import Path
from subprocess import check_output
from tempfile import NamedTemporaryFile

import pandas as pd

from .common import parse_dataframe_nvtx_context


def nsys_to_dataframe(profile: Path | str, use_temp_sql_file: bool = False):
    def refresh_and_load_sql(sql_file: Path, profile: Path, run_args: list[str]):
        if not sql_file.exists():
            check_output(run_args)
        elif sql_file.stat().st_mtime < profile.stat().st_mtime:
            print(f"Regenerating SQL file {sql_file} from profile {profile}.")
            sql_file.unlink()
            check_output(run_args)
        return nsys_sql_to_dataframe(sql_file)

    if not isinstance(profile, Path):
        profile = Path(profile)
    nsys_args = ["nsys", "export", "--type", "sqlite"]
    if not use_temp_sql_file:
        sql_file = profile.with_suffix(".sqlite")
        nsys_args += ["-o", sql_file, profile]
        df = refresh_and_load_sql(sql_file, profile, nsys_args)
    else:
        with NamedTemporaryFile(suffix=".sqlite") as temp_file:
            nsys_args += ["--force-overwrite", "true", "-o", temp_file.name, profile]
            check_output(nsys_args)
            df = nsys_sql_to_dataframe(temp_file.name)
    return parse_dataframe_nvtx_context(df)


def nsys_sql_to_dataframe(sql_file: Path | str):
    conn = sqlite3.connect(sql_file)
    cur = conn.cursor()
    # Set up a view that joins the kernel table, the runtime table, and the NVTX events table.
    # The NVTX events table is used to get the NVTX context for each kernel.
    cur.execute("""DROP VIEW IF EXISTS kernel_with_nvtx;""")
    cur.execute("""
    CREATE VIEW kernel_with_nvtx AS
    SELECT
    k.*,
    max(rt.end, k.end) - rt.start AS totalDuration,
    (
        SELECT e.textId
        FROM NVTX_EVENTS AS e
        WHERE
        e.eventType = 59                  -- Push-pop ranges
        AND e.globalTid = rt.globalTid
        AND e.start   <= rt.start         -- Joins with runtime-event time, not kernel time!
        AND e.end     >= rt.end
        ORDER BY e.start DESC               -- innermost (latest-starting) first
        LIMIT 1
    ) AS nvtxLabel
    FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS rt
    ON k.correlationId = rt.correlationId;
    """)
    # Select a bunch of columns from the view, and exclude any kernels that don't have an NVTX context.
    cur.execute("""
    SELECT
    start AS kernel_start,
    (end - start) / 1000.0 AS latency_us,
    kname.value AS kernel_name,
    nvtxName.value AS ctx_marker,
    registersPerThread,
    staticSharedMemory / 1024 AS staticSharedMemoryKiB,
    dynamicSharedMemory / 1024 AS dynamicSharedMemoryKiB
    FROM kernel_with_nvtx as k
    LEFT JOIN StringIds AS kname ON k.shortName == kname.id
    LEFT JOIN StringIds AS nvtxName ON k.nvtxLabel == nvtxName.id
    WHERE nvtxName.value IS NOT NULL
    ORDER BY kernel_start;
    """)
    df = pd.DataFrame(cur.fetchall(), columns=[col[0] for col in cur.description])
    conn.close()
    return df
