# Base DrivoR Training For MemoryDrivoR

Training commands for the camera-based base DrivoR checkpoints.

## Assumed Setup

Initialize the training and NAVSIMv1 evaluation environment:

```bash
conda activate navsim

export NAVSIM_DEVKIT_ROOT="$MEMORYDRIVOR_ROOT/navsim1"
export OPENSCENE_DATA_ROOT="$MEMORYDRIVOR_ROOT/dataset"
export NAVSIM_EXP_ROOT="$MEMORYDRIVOR_ROOT/exp"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1

cd "$NAVSIM_DEVKIT_ROOT"
```

Place trained checkpoints at:

```text
../weights/original_ckpts/base_drivor_navsim1.pth
../weights/original_ckpts/base_drivor_navsim2.pth
```

## Cache Training Metrics

Cache train metrics:

```bash
python -m navsim.planning.script.run_train_metric_caching \
  train_test_split=navtrain \
  cache.cache_path="$NAVSIM_EXP_ROOT/train_metric_cache"

export TRAIN_METRIC_CACHE_PATH="$NAVSIM_EXP_ROOT/train_metric_cache"
```

For a no-Pittsburgh training ablation, switch the split and cache path:

```bash
python -m navsim.planning.script.run_train_metric_caching \
  train_test_split=navtrain_no_pittsburgh \
  cache.cache_path="$NAVSIM_EXP_ROOT/train_metric_cache_navtrain_no_pittsburgh"

export TRAIN_METRIC_CACHE_PATH="$NAVSIM_EXP_ROOT/train_metric_cache_navtrain_no_pittsburgh"
```

## Train Base DrivoR For NAVSIMv1

Camera-based NAVSIMv1 base checkpoint.

```bash
export EXPERIMENT=base_drivor/navsim1
export NUM_GPUS=4
export GLOBAL_BATCH_SIZE=64
export BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NUM_GPUS))

python -m navsim.planning.script.run_training_full \
  agent=drivoR \
  experiment_name="$EXPERIMENT" \
  train_test_split=navtrain \
  cache_path=null \
  use_cache_without_dataset=false \
  trainer.params.max_epochs=25 \
  trainer.params.accelerator=gpu \
  +trainer.params.devices="$NUM_GPUS" \
  dataloader.params.prefetch_factor=1 \
  dataloader.params.batch_size="$BATCH_SIZE" \
  dataloader.params.num_workers=1 \
  agent.lr_args.name=AdamW \
  agent.lr_args.base_lr=2e-4 \
  agent.num_gpus="$NUM_GPUS" \
  agent.progress_bar=false \
  agent.config.refiner_ls_values=0.0 \
  agent.config.image_backbone.focus_front_cam=false \
  agent.config.one_token_per_traj=true \
  agent.config.refiner_num_heads=1 \
  agent.config.tf_d_model=256 \
  agent.config.tf_d_ffn=1024 \
  agent.config.area_pred=false \
  agent.config.agent_pred=false \
  agent.config.ref_num=4 \
  agent.loss.prev_weight=0.0 \
  agent.config.long_trajectory_additional_poses=2 \
  seed=2
```

After training, copy the selected checkpoint:

```bash
cp "$NAVSIM_EXP_ROOT/ke/$EXPERIMENT"/*/lightning_logs/version_0/checkpoints/last.ckpt \
  "$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_navsim1.pth"
```

Output path: `weights/original_ckpts/base_drivor_navsim1.pth`.

## Train Base DrivoR For NAVSIMv2

Camera-based NAVSIMv2 base checkpoint.

```bash
export EXPERIMENT=base_drivor/navsim2
export NUM_GPUS=4
export GLOBAL_BATCH_SIZE=64
export BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NUM_GPUS))

python -m navsim.planning.script.run_training \
  agent=drivoR \
  experiment_name="$EXPERIMENT" \
  train_test_split=navtrain \
  cache_path=null \
  use_cache_without_dataset=false \
  trainer.params.max_epochs=10 \
  trainer.params.accelerator=gpu \
  +trainer.params.devices="$NUM_GPUS" \
  dataloader.params.prefetch_factor=1 \
  dataloader.params.batch_size="$BATCH_SIZE" \
  dataloader.params.num_workers=1 \
  agent.lr_args.name=AdamW \
  agent.lr_args.base_lr=2e-4 \
  agent.num_gpus="$NUM_GPUS" \
  agent.progress_bar=false \
  agent.config.refiner_ls_values=0.0 \
  agent.config.image_backbone.focus_front_cam=false \
  agent.config.one_token_per_traj=true \
  agent.config.refiner_num_heads=1 \
  agent.config.tf_d_model=256 \
  agent.config.tf_d_ffn=1024 \
  agent.config.area_pred=false \
  agent.config.agent_pred=false \
  agent.config.ref_num=4 \
  agent.loss.prev_weight=0.0 \
  seed=2
```

After training, copy the selected checkpoint:

```bash
cp "$NAVSIM_EXP_ROOT/ke/$EXPERIMENT"/*/lightning_logs/version_0/checkpoints/last.ckpt \
  "$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_navsim2.pth"
```

Output path: `weights/original_ckpts/base_drivor_navsim2.pth`.

## Geographic Training Ablation

For the no-Pittsburgh training ablation, change the training split in the base
training commands:

```bash
train_test_split=navtrain_no_pittsburgh
```

Use a separate experiment name and matching no-Pittsburgh metric cache path.

## Released Checkpoints

Download the released base checkpoints directly to:

```bash
curl -L \
  https://github.com/valeoai/DrivoR/releases/download/model_weights/drivor_Nav1_25epochs.pth \
  -o "$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_navsim1.pth"

curl -L \
  https://github.com/valeoai/DrivoR/releases/download/model_weights/drivor_Nav2_10epochs.pth \
  -o "$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_navsim2.pth"
```

Continue with memory-bank construction in the root `README.md`.
