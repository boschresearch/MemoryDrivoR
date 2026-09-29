# NAVSIM Setup

This guide covers the local folders, environment variables, conda environments, data downloads, and external checkpoints needed for the NAVSIM reproduction workflows.

## Repository Layout

```text
MemoryDrivoR/
|-- docs/navsim/
|-- navsim1/
|-- navsim2/
|-- dataset/        # local NAVSIM and nuPlan-map data
|-- exp/            # local caches, logs, and evaluations
|-- weights/        # local checkpoints
|-- memory_banks/   # local memory-bank files
```

Create the local large-file folders:

```bash
export MEMORYDRIVOR_ROOT="$(pwd)"

mkdir -p \
  "$MEMORYDRIVOR_ROOT/dataset" \
  "$MEMORYDRIVOR_ROOT/dataset/cache" \
  "$MEMORYDRIVOR_ROOT/exp" \
  "$MEMORYDRIVOR_ROOT/weights/original_ckpts" \
  "$MEMORYDRIVOR_ROOT/weights/vit_small_patch14_reg4_dinov2.lvd142m" \
  "$MEMORYDRIVOR_ROOT/memory_banks"
```

## Environment Variables

Use these variables for all NAVSIM commands:

```bash
export MEMORYDRIVOR_ROOT="$(pwd)"
export OPENSCENE_DATA_ROOT="$MEMORYDRIVOR_ROOT/dataset"
export NAVSIM_EXP_ROOT="$MEMORYDRIVOR_ROOT/exp"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export HYDRA_FULL_ERROR=1
```

NAVSIMv1 training, fine-tuning, memory-bank construction, and evaluation:

```bash
export NAVSIM_DEVKIT_ROOT="$MEMORYDRIVOR_ROOT/navsim1"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

NAVSIMv2 evaluation:

```bash
export NAVSIM_DEVKIT_ROOT="$MEMORYDRIVOR_ROOT/navsim2"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

## Install Training And NAVSIMv1 Evaluation Environment

The `navsim` environment is used for NAVSIMv1 training/fine-tuning/evaluation
and NAVSIMv2 training/fine-tuning.

```bash
cd "$MEMORYDRIVOR_ROOT/navsim1"

conda create -n navsim python=3.9 pip=23.3.1 -y
conda activate navsim

python -m pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e ./nuplan-devkit
python -m pip install -e .
```

## Install NAVSIMv2 Evaluation Environment

The NAVSIMv2 evaluation environment is named `navsim2-eval`.

```bash
cd "$MEMORYDRIVOR_ROOT/navsim2"

conda create -n navsim2-eval python=3.9 pip=23.3.1 -y
conda activate navsim2-eval

python -m pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Download NAVSIM Data

Review the nuPlan/OpenScene license. Run download scripts from `dataset/`.

```bash
cd "$OPENSCENE_DATA_ROOT"

# Maps
bash "$MEMORYDRIVOR_ROOT/navsim1/download/download_maps.sh"

# NAVSIM train split used for training and memory-bank construction
bash "$MEMORYDRIVOR_ROOT/navsim1/download/download_navtrain.sh"
mkdir -p navsim_logs sensor_blobs
mv trainval_navsim_logs navsim_logs/trainval
rsync -a trainval_sensor_blobs/trainval/ sensor_blobs/trainval/
rm -rf trainval_sensor_blobs

# NAVSIMv1 test split
bash "$MEMORYDRIVOR_ROOT/navsim1/download/download_test.sh"
mv test_navsim_logs navsim_logs/test
mv test_sensor_blobs sensor_blobs/test

# NAVSIMv2 two-stage evaluation data
bash "$MEMORYDRIVOR_ROOT/navsim2/download/download_navhard_two_stage.sh"
```

Expected data layout:

```text
dataset/
|-- maps/
|-- navsim_logs/
|   |-- trainval/
|   `-- test/
|-- sensor_blobs/
|   |-- trainval/
|   `-- test/
`-- navhard_two_stage/
    |-- sensor_blobs/
    `-- synthetic_scene_pickles/
```

## Download External Checkpoints

The DrivoR image backbone uses the DINOv2 ViT-S register-token checkpoint from
`timm/vit_small_patch14_reg4_dinov2.lvd142m`.

```bash
git lfs install
git clone https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m "$MEMORYDRIVOR_ROOT/weights/vit_small_patch14_reg4_dinov2.lvd142m"
```

The camera-based base DrivoR checkpoints can be downloaded from the original
DrivoR release:

```bash
curl -L https://github.com/valeoai/DrivoR/releases/download/model_weights/drivor_Nav1_25epochs.pth -o "$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_navsim1.pth"

curl -L https://github.com/valeoai/DrivoR/releases/download/model_weights/drivor_Nav2_10epochs.pth -o "$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_navsim2.pth"
```

MemoryDrivoR NAVSIM checkpoints and memory banks use the Zenodo record listed in
the [root README artifact table](../../README.md#artifacts):

```bash
export MEMORYDRIVOR_CLOUD="https://zenodo.org/records/22959058/files"

# Use our reproduced NAVSIMv2 base checkpoint for the paper results.
curl -L "$MEMORYDRIVOR_CLOUD/base_drivor_navsim2.ckpt" \
  -o "$MEMORYDRIVOR_ROOT/weights/original_ckpts/base_drivor_navsim2.pth"

for file in \
  memorydrivor_navsim1.ckpt \
  memorydrivor_navsim2.ckpt \
  ego_only_navsim1.ckpt \
  ego_only_navsim2.ckpt \
  hdmap_only_navsim1.ckpt \
  hdmap_only_navsim2.ckpt
do
  curl -L "$MEMORYDRIVOR_CLOUD/$file" -o "$MEMORYDRIVOR_ROOT/weights/$file"
done

for file in memory_bank_navsim1.pt memory_bank_navsim2.pt
do
  curl -L "$MEMORYDRIVOR_CLOUD/$file" -o "$MEMORYDRIVOR_ROOT/memory_banks/$file"
done
```
