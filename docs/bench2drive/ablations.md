# Bench2Drive Ablations

This guide covers the Town11 and Town13 geographic split evaluations. Complete
the Bench2Drive setup in [setup.md](setup.md) first, then run the common
closed-loop setup and DrivoR variant selection from
[reproduction.md](reproduction.md#bench2drive-closed-loop-evaluation).

The commands below reuse `BENCH2DRIVE_DEVKIT_ROOT`, `Bench2Drive_ROOT`,
`NAVSIM_EXP_ROOT`, `RUN_NAME`, and `CHECKPOINT` from the closed-loop evaluation
setup. The split route files are generated under `NAVSIM_EXP_ROOT`; do not add
them to the official Bench2Drive clone.

## Geographic Split: Town11 And Town13

Evaluate only Town11:

```bash
export ROUTE_DIR="$NAVSIM_EXP_ROOT/b2d_routes"
export ROUTES="$ROUTE_DIR/bench2drive_town11.xml"
export SAVE_PATH="$NAVSIM_EXP_ROOT/b2d_eval/$RUN_NAME/town11"
export CHECKPOINT_ENDPOINT="$SAVE_PATH/eval_bench2drive_town11.json"
mkdir -p "$ROUTE_DIR" "$SAVE_PATH"

python -u "$BENCH2DRIVE_DEVKIT_ROOT/scripts/b2d/filter_routes_by_town.py" \
  --input "$Bench2Drive_ROOT/leaderboard/data/bench2drive220.xml" \
  --town Town11 \
  --output "$ROUTES"

python -u "$BENCH2DRIVE_DEVKIT_ROOT/scripts/b2d/run_b2d_eval.py" \
  --checkpoint "$CHECKPOINT" \
  --routes "$ROUTES" \
  --save-path "$SAVE_PATH" \
  --checkpoint-endpoint "$CHECKPOINT_ENDPOINT" \
  --port 20000 \
  --traffic-manager-port 20500 \
  --gpu-rank 0 \
  --timeout 600.0
```

Evaluate only Town13:

```bash
export ROUTE_DIR="$NAVSIM_EXP_ROOT/b2d_routes"
export ROUTES="$ROUTE_DIR/bench2drive_town13.xml"
export SAVE_PATH="$NAVSIM_EXP_ROOT/b2d_eval/$RUN_NAME/town13"
export CHECKPOINT_ENDPOINT="$SAVE_PATH/eval_bench2drive_town13.json"
mkdir -p "$ROUTE_DIR" "$SAVE_PATH"

python -u "$BENCH2DRIVE_DEVKIT_ROOT/scripts/b2d/filter_routes_by_town.py" \
  --input "$Bench2Drive_ROOT/leaderboard/data/bench2drive220.xml" \
  --town Town13 \
  --output "$ROUTES"

python -u "$BENCH2DRIVE_DEVKIT_ROOT/scripts/b2d/run_b2d_eval.py" \
  --checkpoint "$CHECKPOINT" \
  --routes "$ROUTES" \
  --save-path "$SAVE_PATH" \
  --checkpoint-endpoint "$CHECKPOINT_ENDPOINT" \
  --port 20000 \
  --traffic-manager-port 20500 \
  --gpu-rank 0 \
  --timeout 600.0
```

The evaluator writes route records and global scores to `CHECKPOINT_ENDPOINT`.
Report the split-specific scores directly for the geographic ablation. If you
evaluate route shards separately and need an aggregate benchmark score, merge
them with the upstream Bench2Drive tools before reporting metrics.
