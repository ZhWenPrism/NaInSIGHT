#!/usr/bin/env python3
"""Stage 1 of cohort construction: structural eligibility from the admission registry."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from nainsight.data.cohort import assign_internal_ids, structural_eligibility
from nainsight.data.registry import load_registry
from nainsight.utils import environment_report, get_logger, save_json, seed_everything, timer
from nainsight.workspace import common_parser, load_run_context


def main() -> None:
    ap = common_parser(__doc__)
    ap.add_argument("--site", default="development")
    ap.add_argument("--limit-stays", type=int, default=None, help="random subset for smoke tests")
    args = ap.parse_args()
    cfg, ws = load_run_context(args)
    seed_everything(cfg.seed)
    site = ws.source(args.site)
    out = ws.dir("interim", args.site)
    log = get_logger("cohort", out / "build_cohort.log")
    with timer(log, "registry"):
        reg, schema = load_registry(site, cache_dir=out)
    log.info("registry admissions: %d (patients %d)", len(reg), reg["subject_key"].nunique())
    stage1, flow = structural_eligibility(reg, cfg, site)
    if args.limit_stays and len(stage1) > args.limit_stays:
        keep = np.random.default_rng(cfg.seed).choice(len(stage1), args.limit_stays, replace=False)
        stage1 = stage1.iloc[np.sort(keep)].reset_index(drop=True)
        flow.add(f"smoke-test subset ({args.limit_stays})", stage1)
    stage1, crosswalk = assign_internal_ids(stage1)
    stage1.drop(columns=["stay_key", "subject_key"]).to_parquet(out / "cohort_stage1.parquet")
    crosswalk.to_parquet(out / "_crosswalk.parquet")          # local only: links internal ids to source keys
    flow.frame().to_csv(out / "cohort_flow_stage1.csv", index=False)
    save_json({"static_schema": schema, "environment": environment_report()}, out / "registry_schema.json")
    log.info("\n%s", flow.frame().to_string(index=False))


if __name__ == "__main__":
    main()
