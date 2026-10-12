#!/usr/bin/env bash
# Full NaInSIGHT pipeline.  Usage: scripts/run_pipeline.sh [extra --set overrides...]
#   SMOKE=1 scripts/run_pipeline.sh   -> small subsets and one short epoch (end-to-end check)
set -eo pipefail          # no -u: empty arrays are unbound in bash 3.2
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
EXTRA=("$@")
if [[ "${SMOKE:-0}" == "1" ]]; then
  LIMIT=(--limit-stays 600); REF_LIMIT=(--limit-stays 3000)
  TRAIN_SET=(train.epochs=1 train.max_steps_per_epoch=10 evaluate.bootstrap=20 graph.gnn.epochs=50)
else
  LIMIT=(); REF_LIMIT=(); TRAIN_SET=()
fi
$PY scripts/build_cohort.py "${LIMIT[@]}" --set "${EXTRA[@]}"
$PY scripts/build_events.py --set "${EXTRA[@]}"
$PY scripts/build_landmarks.py --set "${EXTRA[@]}"
$PY scripts/build_reference_graph.py "${REF_LIMIT[@]}" --set "${EXTRA[@]}"
$PY scripts/pretrain_graph_prior.py --set "${TRAIN_SET[@]}" "${EXTRA[@]}"
$PY scripts/build_tensors.py --set "${EXTRA[@]}"
$PY scripts/train.py --run-name nainsight --set "${TRAIN_SET[@]}" "${EXTRA[@]}"
$PY scripts/evaluate.py --run-name nainsight --splits development_test --set "${TRAIN_SET[@]}" "${EXTRA[@]}"
if [[ -f configs/local/external.yaml ]]; then
  $PY scripts/build_cohort.py --site external --set "${EXTRA[@]}"
  $PY scripts/build_events.py --site external --set "${EXTRA[@]}"
  $PY scripts/build_landmarks.py --site external --set "${EXTRA[@]}"
  $PY scripts/build_tensors.py --site external --set "${EXTRA[@]}"
  $PY scripts/evaluate.py --run-name nainsight --splits development_test external_external --set "${EXTRA[@]}"
fi
