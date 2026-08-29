# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import re


def normalize_log_group(log_name: str) -> str:
    """
    Normalize split log segment names to a source-log group.
    Example:
      2021.08.31.12.21.30_veh-40_00378_00527 -> 2021.08.31.12.21.30_veh-40
    """
    log_name = str(log_name)
    match = re.match(r"^(.*)_\d+_\d+$", log_name)
    return match.group(1) if match else log_name

