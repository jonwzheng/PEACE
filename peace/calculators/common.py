from __future__ import annotations

import re
from typing import Optional

HARTREE_TO_KCAL_MOL = 627.5094740631
EV_TO_KCAL_MOL = 23.06054783061903

# Default thermochemical / Boltzmann temperature (Kelvin).
DEFAULT_TEMPERATURE_K = 298.15

# xTB --opt convergence presets, loosest to tightest.
OPT_CONVERGENCE_LEVELS: tuple[str, ...] = (
    "crude",
    "sloppy",
    "loose",
    "lax",
    "normal",
    "tight",
    "vtight",
    "extreme",
)


def looser_opt_convergence_level(opt_level: str) -> Optional[str]:
    """Return the next-looser convergence preset, or None if already at crude."""
    try:
        idx = OPT_CONVERGENCE_LEVELS.index(opt_level)
    except ValueError:
        return None
    if idx == 0:
        return None
    return OPT_CONVERGENCE_LEVELS[idx - 1]


def opt_convergence_retry_levels(opt_level: str) -> list[str]:
    """Return convergence presets to try, from opt_level down to crude."""
    try:
        idx = OPT_CONVERGENCE_LEVELS.index(opt_level)
    except ValueError:
        return [opt_level]
    return list(OPT_CONVERGENCE_LEVELS[idx::-1])


def float_regex() -> str:
    return r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"


def parse_last_float(patterns: list[str], text: str) -> Optional[float]:
    for pat in patterns:
        matches = list(re.finditer(pat, text, flags=re.IGNORECASE | re.MULTILINE))
        if not matches:
            continue
        for m in reversed(matches):
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None
