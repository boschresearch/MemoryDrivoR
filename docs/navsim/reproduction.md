# NAVSIM Reproduction

This guide covers NAVSIMv1 and NAVSIMv2 base checkpoints, memory-bank
construction, MemoryDrivoR fine-tuning, ego-only fine-tuning, and NAVSIMv1
evaluation.
Complete the NAVSIM setup in [setup.md](setup.md) first.

Prebuilt MemoryDrivoR checkpoints and memory banks will be released soon. Until
then, the commands below create the files needed for evaluation.

The `navsim1/` codebase is used for all NAVSIM training and fine-tuning,
including NAVSIMv2 models, and for NAVSIMv1 evaluation. The `navsim2/` codebase
is used only for NAVSIMv2 evaluation.

## Train Base DrivoR Checkpoints

Train the camera-based base DrivoR checkpoints with `navsim1/README.md`, then
place the checkpoints at:

```text
weights/original_ckpts/base_drivor_navsim1.pth
weights/original_ckpts/base_drivor_navsim2.pth
```

The original DrivoR download commands in the setup guide place these files at
the same paths.

## Construct NAVSIM Memory Banks

Memory banks are constructed with the frozen base DrivoR checkpoint. The bank
stores scene tokens, global poses, log names, and precomputed train neighbors.

```bash
conda activate navsim

export NAVSIM_DEVKIT_ROOT="$MEMORYDRIVOR_ROOT/navsim1"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$NAVSIM_DEVKIT_ROOT"

python -m navsim.planning.script.run_build_episodic_memory_bank \
  agent=drivoR_epi_mem \
  experiment_name=episodic_bank/navtrain/navsim1 \
  +include_val_logs_in_bank=true \
  cache_path=null \
  use_cache_without_dataset=false \
  dataloader.params.batch_size=16 \
  dataloader.params.num_workers=2 \
  dataloader.params.prefetch_factor=1 \
  agent.checkpoint_path="$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_navsim1.pth" \
  agent.config.epi_memory.bank.path="$MEMORYDRIVOR_ROOT/memory_banks/memory_bank_navsim1.pt" \
  agent.config.epi_memory.bank.top_k=50
```

For NAVSIMv2, use the NAVSIMv2 base checkpoint and output path:

```bash
python -m navsim.planning.script.run_build_episodic_memory_bank \
  agent=drivoR_epi_mem \
  experiment_name=episodic_bank/navtrain/navsim2 \
  +include_val_logs_in_bank=true \
  cache_path=null \
  use_cache_without_dataset=false \
  dataloader.params.batch_size=16 \
  dataloader.params.num_workers=2 \
  dataloader.params.prefetch_factor=1 \
  agent.checkpoint_path="$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_navsim2.pth" \
  agent.config.epi_memory.bank.path="$MEMORYDRIVOR_ROOT/memory_banks/memory_bank_navsim2.pt" \
  agent.config.epi_memory.bank.top_k=50
```

The builders write:
`memory_banks/memory_bank_navsim1.pt` and `memory_banks/memory_bank_navsim2.pt`.

## Fine-Tune MemoryDrivoR On The Memory Bank

Set the version-specific paths:

```bash
conda activate navsim

export NAVSIM_DEVKIT_ROOT="$MEMORYDRIVOR_ROOT/navsim1"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$NAVSIM_DEVKIT_ROOT"

export BASE_CKPT="$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_navsim1.pth"
export BANK_PATH="$MEMORYDRIVOR_ROOT/memory_banks/memory_bank_navsim1.pt"
export LONG_TRAJ_ADD_POSES=2
export EXPERIMENT="finetuning_e2e/navtrain/memorydrivor_navsim1"
```

For NAVSIMv2 fine-tuning, use:

```bash
export BASE_CKPT="$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_navsim2.pth"
export BANK_PATH="$MEMORYDRIVOR_ROOT/memory_banks/memory_bank_navsim2.pt"
export LONG_TRAJ_ADD_POSES=-1
export EXPERIMENT="finetuning_e2e/navtrain/memorydrivor_navsim2"
```

Then run:

```bash
export NUM_GPUS=4
export GLOBAL_BATCH_SIZE=64
export BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NUM_GPUS))

python -m navsim.planning.script.run_training_full \
  train_ckpt_path="$BASE_CKPT" \
  load_state=false \
  strict_load=false \
  agent=drivoR_epi_mem \
  experiment_name="$EXPERIMENT" \
  cache_path=null \
  use_cache_without_dataset=false \
  trainer.params.max_epochs=5 \
  +trainer.params.devices="$NUM_GPUS" \
  dataloader.params.prefetch_factor=1 \
  dataloader.params.batch_size="$BATCH_SIZE" \
  dataloader.params.num_workers=1 \
  agent.lr_args.base_lr=2e-4 \
  agent.num_gpus="$NUM_GPUS" \
  agent.progress_bar=false \
  agent.config.long_trajectory_additional_poses="$LONG_TRAJ_ADD_POSES" \
  agent.config.skip_perception_backbone=true \
  agent.config.cam_f0=[] \
  agent.config.cam_l0=[] \
  agent.config.cam_r0=[] \
  agent.config.cam_b0=[] \
  agent.config.epi_memory.bank.path="$BANK_PATH" \
  seed=2
```

Use the trained checkpoints at:
`weights/memorydrivor_navsim1.ckpt` and `weights/memorydrivor_navsim2.ckpt`.

## Fine-Tune DrivoR Only On Ego Status

No-camera/no-memory control on `navtrain`.

```bash
conda activate navsim

export NAVSIM_DEVKIT_ROOT="$MEMORYDRIVOR_ROOT/navsim1"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$NAVSIM_DEVKIT_ROOT"

export BASE_CKPT="$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_navsim1.pth"
export LONG_TRAJ_ADD_POSES=2
export EXPERIMENT="finetuning_ego_only/navtrain/navsim1"
export NUM_GPUS=4
export GLOBAL_BATCH_SIZE=64
export BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NUM_GPUS))

python -m navsim.planning.script.run_training_full \
  experiment_name="$EXPERIMENT" \
  cache_path=null \
  use_cache_without_dataset=false \
  trainer.params.max_epochs=5 \
  +trainer.params.devices="$NUM_GPUS" \
  dataloader.params.prefetch_factor=1 \
  dataloader.params.batch_size="$BATCH_SIZE" \
  dataloader.params.num_workers=1 \
  agent.lr_args.base_lr=2e-4 \
  agent.num_gpus="$NUM_GPUS" \
  agent.progress_bar=false \
  agent.config.long_trajectory_additional_poses="$LONG_TRAJ_ADD_POSES" \
  agent.config.freeze_backbone=true \
  agent.config.skip_perception_backbone=true \
  agent.config.cam_f0=[] \
  agent.config.cam_l0=[] \
  agent.config.cam_r0=[] \
  agent.config.cam_b0=[] \
  train_ckpt_path="$BASE_CKPT" \
  load_state=false \
  seed=2
```

For the NAVSIMv2 ego-only control, switch `BASE_CKPT`,
`LONG_TRAJ_ADD_POSES=-1`, and `EXPERIMENT=finetuning_ego_only/navtrain/navsim2`.

Use the trained checkpoints at:
`weights/ego_only_navsim1.ckpt` and `weights/ego_only_navsim2.ckpt`.

## NAVSIMv1 Evaluation

Build the NAVSIMv1 metric cache:

```bash
conda activate navsim

export NAVSIM_DEVKIT_ROOT="$MEMORYDRIVOR_ROOT/navsim1"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$NAVSIM_DEVKIT_ROOT"

python -m navsim.planning.script.run_metric_caching \
  train_test_split=navtest \
  cache.cache_path="$NAVSIM_EXP_ROOT/test_metric_cache_navtest"
```

Evaluate MemoryDrivoR on `navtest`:

```bash
export CHECKPOINT="$MEMORYDRIVOR_ROOT/weights/memorydrivor_navsim1.ckpt"
export BANK_PATH="$MEMORYDRIVOR_ROOT/memory_banks/memory_bank_navsim1.pt"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"

python -m navsim.planning.script.run_pdm_score_multi_gpu \
  experiment_name=eval/navsim1/memorydrivor \
  metric_cache_path="$NAVSIM_EXP_ROOT/test_metric_cache_navtest" \
  agent=drivoR_epi_mem \
  agent.checkpoint_path="$CHECKPOINT" \
  agent.config.long_trajectory_additional_poses=2 \
  agent.config.skip_perception_backbone=true \
  agent.config.epi_memory.bank.path="$BANK_PATH" \
  agent.config.epi_memory.bank.use_precomputed_train_neighbors=false
```

For the ego-only control, use `agent=drivoR`,
`agent.checkpoint_path="$MEMORYDRIVOR_ROOT/weights/ego_only_navsim1.ckpt"`, the
no-camera overrides, and no `agent.config.epi_memory.*` overrides.
