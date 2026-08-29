# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_zoo_root = Path(os.environ.get("Bench2DriveZoo_ROOT", _repo_root / "Bench2DriveZoo")).expanduser()
_external_package = _zoo_root / "mmcv"
_external_init = _external_package / "__init__.py"

if not _external_init.is_file():
    raise ModuleNotFoundError(
        f"Bench2DriveZoo mmcv package not found at {_external_package}. "
        "Set Bench2DriveZoo_ROOT to the official Bench2DriveZoo clone."
    )

__file__ = str(_external_init)
__path__ = [str(_external_package)]

exec(compile(_external_init.read_bytes(), __file__, "exec"), globals())
