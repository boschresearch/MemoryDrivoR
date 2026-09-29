# Bench2Drive Reproduction

This guide covers Bench2Drive base training of DrivoR, memory-bank construction,
MemoryDrivoR fine-tuning, ego-only fine-tuning, and closed-loop evaluation.
Complete the Bench2Drive setup in [setup.md](setup.md) first. Training,
fine-tuning, and memory-bank regeneration also require the Bench2Drive metadata,
cache, and `map.pkl` generation steps. Closed-loop evaluation only needs the
environment, CARLA simulator, checkpoint, memory bank, and DINOv2 backbone.

Some commands invoke `navsim/planning/...` because the Bench2Drive workflow
reuses the DrivoR package layout.

## Train Base DrivoR Checkpoint

Pretrained artifact: download the Bench2Drive base checkpoint from the
[root README artifact table](../../README.md#artifacts) and place it at:

```text
weights/original_ckpts/base_drivor_b2d.ckpt
```

To train it from scratch with 2xH200 on the Bench2Drive Base cache:

```bash
conda activate memorydrivor-b2d

cd "$BENCH2DRIVE_DEVKIT_ROOT"

python navsim/planning/script/run_b2d_training.py \
  experiment_name=b2d/base_drivor_b2d \
  dataloader.params.batch_size=32 \
  dataloader.params.num_workers=8 \
  dataloader.params.prefetch_factor=2 \
  trainer.params.devices=2 \
  agent.num_gpus=2 \
  agent.progress_bar=false \
  strict_load=false \
  seed=2
```
After training, copy the selected
checkpoint to the standardized path:

```bash
cp "$NAVSIM_EXP_ROOT/ke/b2d/base_drivor_b2d/<RUN_ID>/lightning_logs/version_0/checkpoints/last.ckpt" "$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_b2d.ckpt"
```

## Construct The Bench2Drive Memory Bank

Prebuilt artifact: download the Bench2Drive memory bank from the
[root README artifact table](../../README.md#artifacts) and place it at:

```text
memory_banks/memory_bank_b2d.pt
```

To regenerate it, use the Bench2Drive base checkpoint and the Full-set metadata
shards generated in [setup.md](setup.md). The Full dataset is only needed for
this memory-bank construction step.

```bash
conda activate memorydrivor-b2d

cd "$BENCH2DRIVE_DEVKIT_ROOT"

python navsim/planning/script/run_build_b2d_episodic_memory_bank.py \
  --checkpoint-path "$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_b2d.ckpt" \
  --output-path "$MEMORYDRIVOR_ROOT/memory_banks/memory_bank_b2d.pt" \
  --bank-ann-files "$Bench2DriveZoo_ROOT/data/infos-full+base-partial/tmp_data"/b2d_infos_train_*.pkl \
  --query-ann-files \
    "$Bench2DriveZoo_ROOT/data/infos/b2d_infos_train.pkl" \
    "$Bench2DriveZoo_ROOT/data/infos/b2d_infos_val.pkl" \
  --weather-levels easy okay \
  --top-k 20 \
  --batch-size 64 \
  --num-workers 8 \
  --min-translation-m 1.5 \
  --max-interval-s 5 \
  --min-translation-noise-std 0.1
```

The resulting bank is written to `memory_banks/memory_bank_b2d.pt`.

## Fine-Tune MemoryDrivoR On The Bench2Drive Memory Bank

Pretrained artifact: download the Bench2Drive MemoryDrivoR checkpoint from the
[root README artifact table](../../README.md#artifacts) and place it at:

```text
weights/memorydrivor_b2d.ckpt
```

To fine-tune from the Bench2Drive base checkpoint:

```bash
conda activate memorydrivor-b2d

export BASE_CKPT="$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_b2d.ckpt"
export BANK_PATH="$MEMORYDRIVOR_ROOT/memory_banks/memory_bank_b2d.pt"

cd "$BENCH2DRIVE_DEVKIT_ROOT"

python navsim/planning/script/run_b2d_training.py \
  strict_load=false \
  agent=drivoR_b2d_epi_mem \
  experiment_name=b2d/memorydrivor_b2d \
  dataloader.params.batch_size=32 \
  dataloader.params.num_workers=8 \
  dataloader.params.prefetch_factor=1 \
  trainer.params.devices=2 \
  agent.num_gpus=2 \
  agent.progress_bar=false \
  agent.config.enable_collision_gate_for_ep=false \
  agent.config.enable_drivable_gate_for_ep=false \
  agent.config.skip_perception_backbone=true \
  agent.config.epi_memory.bank.path="$BANK_PATH" \
  train_ckpt_path="$BASE_CKPT" \
  seed=2
```

After fine-tuning, copy the selected checkpoint to the standardized path:

```bash
cp "$NAVSIM_EXP_ROOT/ke/b2d/memorydrivor_b2d/<RUN_ID>/lightning_logs/version_0/checkpoints/last.ckpt" \
  "$MEMORYDRIVOR_ROOT/weights/memorydrivor_b2d.ckpt"
```

## Fine-Tune Bench2Drive DrivoR Only On Ego Status

Pretrained artifact: download the Bench2Drive ego-only checkpoint from the
[root README artifact table](../../README.md#artifacts) and place it at:

```text
weights/ego_only_b2d.ckpt
```

To fine-tune the no-camera/no-memory control:

```bash
conda activate memorydrivor-b2d

export BASE_CKPT="$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_b2d.ckpt"

cd "$BENCH2DRIVE_DEVKIT_ROOT"

python navsim/planning/script/run_b2d_training.py \
  experiment_name=b2d/ego_only_b2d \
  dataloader.params.batch_size=32 \
  dataloader.params.num_workers=8 \
  dataloader.params.prefetch_factor=1 \
  trainer.params.devices=2 \
  agent.num_gpus=2 \
  agent.progress_bar=false \
  agent.config.enable_collision_gate_for_ep=false \
  agent.config.enable_drivable_gate_for_ep=false \
  agent.config.skip_perception_backbone=true \
  train_ckpt_path="$BASE_CKPT" \
  seed=2
```

After fine-tuning, copy the selected checkpoint to the standardized path:

```bash
cp "$NAVSIM_EXP_ROOT/ke/b2d/ego_only_b2d/<RUN_ID>/lightning_logs/version_0/checkpoints/last.ckpt" \
  "$MEMORYDRIVOR_ROOT/weights/ego_only_b2d.ckpt"
```

## Bench2Drive Closed-Loop Evaluation


Select which DrivoR variant to evaluate:

```bash
# MemoryDrivoR, used for the Bench2Drive memory result.
export RUN_NAME="memorydrivor_b2d"
export CHECKPOINT="$MEMORYDRIVOR_ROOT/weights/memorydrivor_b2d.ckpt"
export DRIVOR_B2D_AGENT_CONFIG="$BENCH2DRIVE_DEVKIT_ROOT/scripts/b2d/configs/memorydrivor_b2d_ep_only_eval.yaml"

# Ego-only control.
# export RUN_NAME="ego_only_b2d"
# export CHECKPOINT="$MEMORYDRIVOR_ROOT/weights/ego_only_b2d.ckpt"
# export DRIVOR_B2D_AGENT_CONFIG="$BENCH2DRIVE_DEVKIT_ROOT/scripts/b2d/configs/ego_only_b2d_ep_only_eval.yaml"

# Camera-based base DrivoR.
# export RUN_NAME="base_drivor_b2d"
# export CHECKPOINT="$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_b2d.ckpt"
# export DRIVOR_B2D_AGENT_CONFIG="$BENCH2DRIVE_DEVKIT_ROOT/scripts/b2d/configs/base_drivor_b2d_eval.yaml"
```

Evaluate all 220 Bench2Drive routes in a single closed-loop call:

```bash
export SAVE_PATH="$NAVSIM_EXP_ROOT/b2d_eval/$RUN_NAME/all_220"

conda activate memorydrivor-b2d

python -u "$BENCH2DRIVE_DEVKIT_ROOT/scripts/b2d/run_b2d_eval.py" \
  --checkpoint "$CHECKPOINT" \
  --drivor-agent-config "$DRIVOR_B2D_AGENT_CONFIG" \
  --save-path "$SAVE_PATH" \
  --timeout 600.0
```

The evaluator writes route records and global scores to
`$SAVE_PATH/eval_bench2drive220.json`.
For the full benchmark, use the file from `bench2drive220.xml` for Driving Score
and Success Rate reporting. Town11 and Town13 geographic split evaluations are
covered in [ablations.md](ablations.md).
