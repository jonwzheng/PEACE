from __future__ import annotations

import importlib.util
import os
import shutil

import pytest


def _require_or_skip(env_var: str, reason: str) -> None:
    # CI jobs that install the dependency set the env var so a broken install fails loudly.
    if os.environ.get(env_var):
        pytest.fail(f"{reason} ({env_var} is set)")
    pytest.skip(reason)


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("xtb") and shutil.which("xtb") is None:
        _require_or_skip("PEACE_REQUIRE_XTB", "xtb binary not found on PATH")
    if item.get_closest_marker("aimnet2") and importlib.util.find_spec("aimnet") is None:
        _require_or_skip("PEACE_REQUIRE_AIMNET2", "aimnet is not installed (pip install 'aimnet[ase]')")


@pytest.fixture
def no_executable_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let --dry-run solvation tests run on machines without xTB/ORCA installed."""
    import peace.calculators.orca
    import peace.calculators.xtb

    monkeypatch.setattr(peace.calculators.xtb, "verify_xtb_executable", lambda exe, **kwargs: None)
    monkeypatch.setattr(peace.calculators.orca, "verify_orca_executable", lambda exe: exe)
