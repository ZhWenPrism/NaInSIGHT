#!/usr/bin/env python3
"""Reference cohort -> anchor panel -> dose x lag effect spectra (full cohort and neurosurgical sub-population)
-> empirical-Bayes transfer, unit and time-step conversion -> prior matrices and knowledge graph."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from nainsight.graph.build import build_graph
from nainsight.graph.effects import anchor_panel, fit_effects
from nainsight.graph.reference import (reference_drugs, reference_infusions, reference_labs, reference_stays,
                                      reference_urine)
from nainsight.graph.transfer import convert_units, prior_matrices, shrink_subpopulation
from nainsight.utils import environment_report, get_logger, save_json, seed_everything, timer
from nainsight.workspace import common_parser, load_run_context


def main() -> None:
    ap = common_parser(__doc__)
    ap.add_argument("--limit-stays", type=int, default=None, help="random subset of reference stays (smoke tests)")
    ap.add_argument("--reuse-extracts", action="store_true", help="reuse cached reference extracts")
    args = ap.parse_args()
    cfg, ws = load_run_context(args)
    seed_everything(cfg.seed)
    ref = ws.source("reference")
    out = ws.reference
    log = get_logger("reference", out / "build_reference_graph.log")
    cache = out / "extracts"
    cache.mkdir(exist_ok=True)
    if args.reuse_extracts and (cache / "stays.parquet").exists():
        stays, labs, drugs, inf, urine = (pd.read_parquet(cache / f"{n}.parquet")
                                          for n in ("stays", "labs", "drugs", "infusions", "urine"))
    else:
        with timer(log, "reference stays"):
            stays = reference_stays(ref, cfg)
        if args.limit_stays and len(stays) > args.limit_stays:
            stays = stays.sample(args.limit_stays, random_state=cfg.seed).reset_index(drop=True)
        log.info("reference ICU stays: %d (sub-population %d)", len(stays), int(stays["subpop"].sum()))
        with timer(log, "laboratory"):
            labs = reference_labs(ref, stays, cfg)
        with timer(log, "medication administrations"):
            drugs = reference_drugs(ref, stays, cfg)
        with timer(log, "infusions"):
            inf = reference_infusions(ref, stays, cfg)
        with timer(log, "urine output"):
            urine = reference_urine(ref, stays, cfg)
        for name, df in (("stays", stays), ("labs", labs), ("drugs", drugs), ("infusions", inf), ("urine", urine)):
            df.to_parquet(cache / f"{name}.parquet")
    with timer(log, "anchor panel"):
        panel, meta = anchor_panel(stays, labs, drugs, inf, urine, cfg)
    log.info("anchors: %d from %d stays", meta["n_anchors"], meta["n_stays"])
    with timer(log, "effect estimation"):
        eff_all = fit_effects(panel, cfg, "all", cfg.seed)
        eff_sub = fit_effects(panel[panel["subpop"]], cfg, "subpopulation", cfg.seed)
    eff = shrink_subpopulation(eff_all, eff_sub)
    factors = ref.get("unit_to_development") or {}
    eff = convert_units(eff, factors)
    # node descriptors of the state nodes in the same (development) units as the transferred effects
    for key in ("baseline_mean", "baseline_sd"):
        meta[key] = {s: v * float(factors.get(s, 1.0)) for s, v in meta[key].items()}
    pd.concat([eff_all, eff_sub], ignore_index=True).to_parquet(out / "effects_by_stage.parquet")
    eff.to_parquet(out / "effects_transferred.parquet")
    pri = prior_matrices(eff, cfg)
    np.savez(out / "prior_matrices.npz", **pri)
    graph = build_graph(pri, meta)
    np.savez(out / "graph.npz", **graph)
    save_json({"meta": meta, "n_edges": int(graph["edge_index"].shape[1]),
               "environment": environment_report()}, out / "reference_graph_meta.json")
    drug_sodium = eff[(eff["src_type"] == "drug") & (eff["dst"] == "NA") & (eff["lag_h"] == 24)]
    log.info("24-h sodium effects (transferred, development units):\n%s",
             drug_sodium.pivot_table(index="src", columns="level", values="beta").round(3).to_string())


if __name__ == "__main__":
    main()
