# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import sys
from pathlib import Path
from typing import Tuple


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_default_root(env_var: str, local_name: str) -> Path:
    env_value = os.environ.get(env_var)
    if env_value:
        root = Path(env_value).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"{env_var} points to missing path: {root}")
        return root

    repo_local = repo_root() / local_name
    if repo_local.exists():
        return repo_local

    raise FileNotFoundError(
        f"{local_name} not found. Expected either {repo_local} or ${env_var}."
    )


def bench2drive_root() -> Path:
    return _resolve_default_root("Bench2Drive_ROOT", "Bench2Drive")


def bench2drivezoo_root() -> Path:
    return _resolve_default_root("Bench2DriveZoo_ROOT", "Bench2DriveZoo")


def ensure_external_paths() -> Tuple[Path, Path, Path]:
    repo = repo_root()
    b2d_root = bench2drive_root()
    zoo_root = bench2drivezoo_root()

    # Prefer Bench2DriveZoo's bundled ``mmcv`` package.
    paths = [
        zoo_root,
        repo,
        b2d_root,
        b2d_root / "leaderboard",
        b2d_root / "scenario_runner",
    ]
    for path in reversed(paths):
        path_str = str(path)
        while path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)

    return repo, b2d_root, zoo_root
