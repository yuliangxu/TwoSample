#!/usr/bin/env bash
# Submit preparation -> four independent generators -> BATTS/figures/report.
set -euo pipefail
if [[ ! -f scripts/run_ibdmdb_experiment.py ]]; then
  echo 'Run this script from the TwoSample repository root.' >&2
  exit 1
fi
prepare=$(sbatch --parsable hpc/ibdmdb_job.sh prepare)
prepare="${prepare%%;*}"
generators=$(sbatch --parsable --dependency="afterok:$prepare" --array=0-3 hpc/ibdmdb_job.sh generator)
generators="${generators%%;*}"
analysis=$(sbatch --parsable --dependency="afterok:$generators" hpc/ibdmdb_job.sh analysis)
analysis="${analysis%%;*}"
printf 'Preparation: %s\nGenerator array: %s\nBATTS and figures: %s\n' "$prepare" "$generators" "$analysis"
