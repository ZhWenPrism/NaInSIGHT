#!/usr/bin/env python3
"""Event store for stage-1 admissions: laboratory (report time), vital signs, fluid charts, medications."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from nainsight.data.events import extract_io, extract_labs, extract_medications, extract_vitals, stay_windows
from nainsight.utils import get_logger, save_json, timer
from nainsight.workspace import common_parser, load_run_context

EXTRACTORS = {"labs": extract_labs, "vitals": extract_vitals, "io": extract_io, "meds": extract_medications}


def main() -> None:
    ap = common_parser(__doc__)
    ap.add_argument("--site", default="development")
    ap.add_argument("--only", nargs="*", default=list(EXTRACTORS))
    args = ap.parse_args()
    cfg, ws = load_run_context(args)
    site = ws.source(args.site)
    out = ws.dir("interim", args.site)
    log = get_logger("events", out / "build_events.log")
    stays = pd.read_parquet(out / "cohort_stage1.parquet")
    keys = pd.read_parquet(out / "_crosswalk.parquet")[["stay_id", "subject_key"]]
    stays = stays.drop(columns=["subject_key"], errors="ignore").merge(keys, on="stay_id")
    windows = stay_windows(stays, cfg.events.preop_lookback_days, cfg.events.max_postop_h)
    qc = {}
    for name in args.only:
        with timer(log, name):
            df, q = EXTRACTORS[name](site, windows, cfg)
        df.to_parquet(ws.dir("interim", args.site, "events") / f"{name}.parquet")
        qc[name] = q
        log.info("%s: %d rows, %d admissions | %s", name, len(df), df["stay_id"].nunique(),
                 df["concept"].value_counts().head(12).to_dict())
    save_json(qc, out / "events_qc.json")


if __name__ == "__main__":
    main()
