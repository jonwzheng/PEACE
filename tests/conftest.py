from __future__ import annotations

import pytest


@pytest.fixture
def no_executable_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let --dry-run solvation tests run on machines without xTB/ORCA installed."""
    import peace.calculators.orca
    import peace.calculators.xtb

    monkeypatch.setattr(peace.calculators.xtb, "verify_xtb_executable", lambda exe, **kwargs: None)
    monkeypatch.setattr(peace.calculators.orca, "verify_orca_executable", lambda exe: exe)
