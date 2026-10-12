#!/usr/bin/env python3
"""Stage 2 of cohort construction: landmarks, outcomes, evaluability, index admission and splits."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from nainsight.data.cohort import FlowLog, finalize_cohort
from nainsight.data.landmarks import build_landmarks, outcome_composition, preop_hyponatraemia, sodium_events
from nainsight.utils import get_logger
from nainsight.workspace import common_parser, load_run_context


def main() -> None:
    ap = common_parser(__doc__)
    ap.add_argument("--site", default="development")
    ap.add_argument("--role", default=None, help="development (default for the development site) or external")
    args = ap.parse_args()
    cfg, ws = load_run_context(args)
    role = args.role or ("development" if args.site == "development" else "external")
    out = ws.dir("interim", args.site)
    log = get_logger("landmarks", out / "build_landmarks.log")
    stays = pd.read_parquet(out / "cohort_stage1.parquet")
    labs = pd.read_parquet(out / "events" / "labs.parquet")
    na = sodium_events(labs, list(cfg.outcome.sodium_concepts))
    preop = preop_hyponatraemia(stays, na, cfg)
    lmk = build_landmarks(stays, na, cfg)
    flow = FlowLog(args.site)
    flow.rows = pd.read_csv(out / "cohort_flow_stage1.csv").drop(columns=["retained_pct"]).to_dict("records")
    cohort = finalize_cohort(stays, preop, lmk, cfg, flow, cfg.seed, site_role=role)
    lmk = lmk[lmk["stay_id"].isin(cohort["stay_id"])].reset_index(drop=True)
    cohort.to_parquet(out / "cohort.parquet")
    lmk.to_parquet(out / "landmarks.parquet")
    flow.frame().to_csv(out / "cohort_flow.csv", index=False)
    comp = outcome_composition(cohort, lmk, list(cfg.outcome.horizons_h))
    comp.to_csv(out / "outcome_composition.csv", index=False)
    log.info("\n%s", flow.frame().to_string(index=False))
    log.info("splits: %s", cohort["split"].value_counts().to_dict())
    log.info("\n%s", comp.to_string(index=False))


if __name__ == "__main__":
    main()
