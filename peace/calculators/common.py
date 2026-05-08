from __future__ import annotations

import re
from typing import Optional

HARTREE_TO_KCAL_MOL = 627.5094740631


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
