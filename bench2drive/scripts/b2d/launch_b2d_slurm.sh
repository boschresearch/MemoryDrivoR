#!/bin/bash
# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

#SBATCH --job-name=drivor-b2d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

CHECKPOINT=${1:?Usage: $0 /path/to/checkpoint.ckpt TASK_ID /path/to/save_dir}
TASK_ID=${2:?Usage: $0 /path/to/checkpoint.ckpt TASK_ID /path/to/save_dir}
SAVE_DIR=${3:?Usage: $0 /path/to/checkpoint.ckpt TASK_ID /path/to/save_dir}

source ~/.bashrc
conda activate drivoR

export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-$NAVSIM_DEVKIT_ROOT/exp}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-$NAVSIM_DEVKIT_ROOT/dataset}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}
export HYDRA_FULL_ERROR=1

BENCH2DRIVE_ROOT=${Bench2Drive_ROOT:-$NAVSIM_DEVKIT_ROOT/Bench2Drive}
ALGO=${B2D_ALGO_NAME:-drivor}
PLANNER_TYPE=${B2D_PLANNER_TYPE:-closed}
BASE_ROUTE=${B2D_BASE_ROUTE:-$BENCH2DRIVE_ROOT/leaderboard/data/bench2drive220}
SPLIT_ROUTE="${BASE_ROUTE}_${TASK_ID}_${ALGO}_${PLANNER_TYPE}.xml"
if [[ -f "$SPLIT_ROUTE" ]]; then
  ROUTES=$SPLIT_ROUTE
else
  ROUTES="${BASE_ROUTE}.xml"
fi

PORT=${B2D_PORT:-$((20000 + TASK_ID * 150))}
TM_PORT=${B2D_TM_PORT:-$((20500 + TASK_ID * 150))}
GPU_RANK=${B2D_GPU_RANK:-0}

mkdir -p "$SAVE_DIR"

python "$NAVSIM_DEVKIT_ROOT/scripts/b2d/run_b2d_eval.py" \
  --checkpoint "$CHECKPOINT" \
  --save-path "$SAVE_DIR" \
  --routes "$ROUTES" \
  --checkpoint-endpoint "$SAVE_DIR/eval_bench2drive220_${TASK_ID}.json" \
  --port "$PORT" \
  --traffic-manager-port "$TM_PORT" \
  --gpu-rank "$GPU_RANK"
