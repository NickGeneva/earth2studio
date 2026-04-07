# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""earth2bufrio — Pure-Python BUFR decoder for PyArrow Tables."""

from earth2bufrio._api import read_bufr
from earth2bufrio._types import BufrDecodeError

__version__ = "0.1.0"

__all__: list[str] = ["read_bufr", "BufrDecodeError"]
