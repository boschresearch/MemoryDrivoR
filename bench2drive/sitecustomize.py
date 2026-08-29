# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


import sys

try:
    from numba.core import errors as numba_errors
except ImportError:
    pass
else:
    sys.modules.setdefault("numba.errors", numba_errors)