#!/bin/bash
# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

set -euo pipefail

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_b2d_training.py" \
  experiment_name=drivor_b2d \
  agent=drivoR_b2d
