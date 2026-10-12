#!/usr/bin/env python3
"""Fit preprocessors on the training split and encode every split into token/landmark tensors."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from nainsight.data.features import StaticPreprocessor, fit_dose_cutpoints, fit_value_scaler, preop_summaries
from nainsight.data.tokenizer import ClinicalEventTokenizer
from nainsight.utils import environment_report, get_logger, load_json, save_json
from nainsight.workspace import common_parser, load_run_context

EVENT_KINDS = ("labs", "vitals", "io", "meds")


def static_frame(cohort: pd.DataFrame, events: dict, schema: dict) -> tuple[pd.DataFrame, dict]:
    pre = preop_summaries(cohort, events["labs"], events["meds"], events["io"])
    frame = cohort.merge(pre, on="stay_id", how="left").sort_values("stay_id").reset_index(drop=True)
    full = dict(schema)
    for c in pre.columns:
        if c != "stay_id":
            full[c] = "numeric" if (c.startswith("preop_last_") or c.startswith("preop_min_") or "urine" in c) else "binary"
    return frame, full


def main() -> None:
    ap = common_parser(__doc__)
    ap.add_argument("--site", default="development")
    args = ap.parse_args()
    cfg, ws = load_run_context(args)
    site_dir = ws.dir("interim", args.site)
    log = get_logger("tensors", site_dir / "build_tensors.log")
    cohort = pd.read_parquet(site_dir / "cohort.parquet")
    lmk = pd.read_parquet(site_dir / "landmarks.parquet")
    events = {k: pd.read_parquet(site_dir / "events" / f"{k}.parquet") for k in EVENT_KINDS}
    events = {k: v[v["stay_id"].isin(cohort["stay_id"])].reset_index(drop=True) for k, v in events.items()}
    schema = load_json(site_dir / "registry_schema.json")["static_schema"]
    frame, full_schema = static_frame(cohort, events, schema)
    horizons = [int(h) for h in cfg.outcome.horizons_h]
    pre_path = ws.artifacts / "preprocessors.json"

    if args.site == "development":
        train_ids = set(cohort.loc[cohort["split"] == "train", "stay_id"])
        tr_events = {k: v[v["stay_id"].isin(train_ids)] for k, v in events.items()}
        static = StaticPreprocessor(full_schema, cfg.tokens.static_min_category_count).fit(frame[frame["stay_id"].isin(train_ids)])
        scaler = fit_value_scaler(tr_events["labs"], tr_events["vitals"], tr_events["io"], tr_events["meds"])
        cuts = fit_dose_cutpoints(tr_events["meds"], cfg.tokens.dose_levels)
        tok = ClinicalEventTokenizer(cfg.tokens.max_len, cfg.tokens.preop_window_h, cfg.tokens.value_clip,
                                     cfg.tokens.dose_levels, cfg.tokens.min_concept_count).fit(tr_events, scaler, cuts)
        save_json({"tokenizer": tok.state(), "static": static.state(), "horizons": horizons,
                   "environment": environment_report()}, pre_path)
        log.info("preprocessors fitted on %d training admissions: %d concepts, %d ingredients, %d static columns",
                 len(train_ids), len(tok.concept_vocab), len(tok.ingredient_vocab), len(static.columns))
    else:
        pre = load_json(pre_path)
        tok = ClinicalEventTokenizer.from_state(pre["tokenizer"])
        static = StaticPreprocessor.from_state(pre["static"])

    for split in sorted(set(cohort["split"]) - {"unused"}):
        stays = cohort[cohort["split"] == split].sort_values("stay_id")
        rows = frame[frame["stay_id"].isin(stays["stay_id"])].sort_values("stay_id")
        pack = tok.encode(stays, events, lmk, static.transform(rows).values, horizons)
        np.savez(ws.tensors / f"{args.site}_{split}.npz", **pack)
        log.info("%s/%s: %d admissions, %d landmarks, %d tokens (truncated sequences: %d)", args.site, split,
                 len(pack["stay_id"]), len(pack["lmk_time"]), len(pack["tok_concept"]), int(pack["n_truncated"]))


if __name__ == "__main__":
    main()
