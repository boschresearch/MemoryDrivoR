# Ablations

This guide covers the geographic split and HD-map-only ablations from the main
paper. Complete the setup in [setup.md](setup.md) first.

## Geographic Splits

For training-side geographic ablations, replace `train_test_split=navtrain` with
`train_test_split=navtrain_no_pittsburgh` in the base training, memory-bank
construction, and fine-tuning commands.

NAVSIMv2 Pittsburgh evaluation uses the camera-based base DrivoR checkpoint on
`navhard_pittsburgh`:

```bash
conda activate navsim2-eval

export NAVSIM_DEVKIT_ROOT="$MEMORYDRIVOR_ROOT/navsim2"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"
cd "$NAVSIM_DEVKIT_ROOT"

export TRAIN_TEST_SPLIT=navhard_pittsburgh
export CACHE_PATH="$NAVSIM_EXP_ROOT/navhard_two_stage_metric_cache"
export CHECKPOINT="$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_navsim2.pth"

python -m navsim.planning.script.run_pdm_score_gpu_v2 \
  train_test_split="$TRAIN_TEST_SPLIT" \
  experiment_name=eval/navsim2_pittsburgh/base_drivor_camera \
  metric_cache_path="$CACHE_PATH" \
  agent=drivoR \
  agent.checkpoint_path="$CHECKPOINT" \
  agent.config.noc=10 \
  agent.config.dac=13 \
  agent.config.ddc=6 \
  agent.config.ttc=14 \
  agent.config.ep=15
```

To evaluate the no-Pittsburgh geographic split with the same camera-based base
checkpoint, set `TRAIN_TEST_SPLIT=navhard_no_pittsburgh`. For the full NAVSIMv2
hard split, set `TRAIN_TEST_SPLIT=navhard_two_stage`.

## HD-Map-Only Input Source

For the HD-map-only ablation, cache map-only features and training metric
targets:

```bash
conda activate navsim

export NAVSIM_DEVKIT_ROOT="$MEMORYDRIVOR_ROOT/navsim1"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$NAVSIM_DEVKIT_ROOT"

export HDMAP_CACHE_PATH="$OPENSCENE_DATA_ROOT/cache/navsim_cache_hdmap_navtrain"
export HDMAP_TRAIN_METRIC_CACHE="$OPENSCENE_DATA_ROOT/cache/train_metric_cache_hdmap_navtrain"
export TRAIN_METRIC_CACHE_PATH="$HDMAP_TRAIN_METRIC_CACHE"

python -m navsim.planning.script.run_dataset_caching \
  worker=single_machine_thread_pool \
  worker.max_workers=24 \
  train_test_split=navtrain \
  cache_path="$HDMAP_CACHE_PATH" \
  force_cache_computation=false \
  experiment_name=navsim_cache_hdmap_navtrain \
  agent.config.hd_map.enabled=true \
  agent.config.skip_perception_backbone=true \
  agent.config.epi_memory.enabled=false

python -m navsim.planning.script.run_train_metric_caching \
  train_test_split=navtrain \
  cache.cache_path="$HDMAP_TRAIN_METRIC_CACHE"
```

Then fine-tune HD-map-only DrivoR:

```bash
export BASE_CKPT="$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_navsim1.pth"
export NUM_GPUS=4
export GLOBAL_BATCH_SIZE=64
export BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NUM_GPUS))

python -m navsim.planning.script.run_training_full \
  agent=drivoR \
  experiment_name=finetuning_e2e/navtrain/hdmap_only_navsim1 \
  train_test_split=navtrain \
  cache_path="$HDMAP_CACHE_PATH" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  trainer.params.max_epochs=5 \
  trainer.params.accelerator=gpu \
  +trainer.params.devices="$NUM_GPUS" \
  dataloader.params.prefetch_factor=1 \
  dataloader.params.batch_size="$BATCH_SIZE" \
  dataloader.params.num_workers=1 \
  agent.lr_args.name=AdamW \
  agent.lr_args.base_lr=2e-4 \
  agent.num_gpus="$NUM_GPUS" \
  agent.progress_bar=false \
  agent.config.hd_map.enabled=true \
  agent.config.skip_perception_backbone=true \
  agent.config.epi_memory.enabled=false \
  agent.config.cam_f0=[] \
  agent.config.cam_l0=[] \
  agent.config.cam_l1=[] \
  agent.config.cam_l2=[] \
  agent.config.cam_r0=[] \
  agent.config.cam_r1=[] \
  agent.config.cam_r2=[] \
  agent.config.cam_b0=[] \
  agent.config.long_trajectory_additional_poses=2 \
  train_ckpt_path="$BASE_CKPT" \
  load_state=false \
  strict_load=false \
  seed=2
```

For NAVSIMv2 HD-map-only evaluation, use the NAVSIMv2 evaluator command with
`agent=drivoR`, the HD-map-only checkpoint, no episodic-memory overrides, and:

```bash
agent.config.hd_map.enabled=true
agent.config.skip_perception_backbone=true
```

If the artifact download commands from [setup.md](setup.md) were run, these
files are already in place.
