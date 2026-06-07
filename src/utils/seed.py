"""Random seed helper."""

from __future__ import annotations

import random


def set_seed(seed: int) -> None:
    """Set standard-library seed.

    NumPy/PyTorch seed hooks will be added when those dependencies are used.
    """

    random.seed(seed)
