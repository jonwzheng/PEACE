"""Solvent validation for combined ALPB optimization and CPCM-X solvation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ALPB_SOLVENTS_FILE = _PACKAGE_ROOT / "data" / "allowed_solvents.txt"
CPCM_SOLVENTS_FILE = _PACKAGE_ROOT / "data" / "cpcm_allowed_solvents.txt"

# ALPB names that must not be exposed even when a CPCM alias exists.
_BLOCKED_ALPB_SOLVENTS = frozenset({"ch2cl2", "chcl3", "ether", "woctanol", "hexandecane"})


@dataclass(frozen=True)
class SolventNames:
    """Canonical ALPB and CPCM-X solvent names for one supported phase."""

    alpb: str
    cpcm: str


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def _parse_solvent_list(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Solvent list not found: {path}")
    solvents: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip().lower()
        if name and not name.startswith("#"):
            solvents.append(name)
    if not solvents:
        raise ValueError(f"No solvents listed in {path}")
    return tuple(solvents)


@lru_cache(maxsize=1)
def load_alpb_solvents(path: Path | None = None) -> tuple[str, ...]:
    return _parse_solvent_list(path or ALPB_SOLVENTS_FILE)


@lru_cache(maxsize=1)
def load_cpcm_solvent_entries(path: Path | None = None) -> tuple[tuple[str, frozenset[str]], ...]:
    solvent_file = path or CPCM_SOLVENTS_FILE
    if not solvent_file.is_file():
        raise FileNotFoundError(f"CPCM solvent list not found: {solvent_file}")

    entries: list[tuple[str, frozenset[str]]] = []
    for line in solvent_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        aliases = [part.strip() for part in re.split(r"\s*/\s*", stripped) if part.strip()]
        if not aliases:
            continue
        keys = frozenset(_normalize_key(alias) for alias in aliases)
        entries.append((aliases[0], keys))
    if not entries:
        raise ValueError(f"No CPCM solvents listed in {solvent_file}")
    return tuple(entries)


def _cpcm_name_for_alpb(alpb_name: str, cpcm_entries: tuple[tuple[str, frozenset[str]], ...]) -> str | None:
    key = _normalize_key(alpb_name)
    for primary_name, alias_keys in cpcm_entries:
        if key in alias_keys:
            return primary_name
    return None


@lru_cache(maxsize=1)
def load_supported_solvents() -> tuple[SolventNames, ...]:
    """Return solvents supported by both ALPB and CPCM-X."""
    cpcm_entries = load_cpcm_solvent_entries()
    supported: list[SolventNames] = []
    missing_cpcm: list[str] = []

    for alpb_name in load_alpb_solvents():
        if alpb_name in _BLOCKED_ALPB_SOLVENTS:
            continue
        cpcm_name = _cpcm_name_for_alpb(alpb_name, cpcm_entries)
        if cpcm_name is None:
            missing_cpcm.append(alpb_name)
            continue
        supported.append(SolventNames(alpb=alpb_name, cpcm=cpcm_name))

    if missing_cpcm:
        raise ValueError(
            "ALPB solvents without a CPCM-X mapping: "
            + ", ".join(sorted(missing_cpcm))
        )
    if not supported:
        raise ValueError("No solvents are supported by both ALPB and CPCM-X.")
    return tuple(supported)


@lru_cache(maxsize=1)
def _alias_lookup() -> dict[str, SolventNames]:
    lookup: dict[str, SolventNames] = {}
    cpcm_entries = load_cpcm_solvent_entries()
    for spec in load_supported_solvents():
        lookup[_normalize_key(spec.alpb)] = spec
        for _primary_name, alias_keys in cpcm_entries:
            if _normalize_key(spec.cpcm) in alias_keys or spec.cpcm == _primary_name:
                if _primary_name == spec.cpcm:
                    for alias_key in alias_keys:
                        lookup.setdefault(alias_key, spec)
                    break
    return lookup


def resolve_solvent(name: str) -> SolventNames:
    """Resolve a user solvent name to canonical ALPB and CPCM-X names."""
    key = _normalize_key(name)
    if not key:
        raise ValueError("Solvent name must not be empty.")

    spec = _alias_lookup().get(key)
    if spec is not None:
        return spec

    allowed = ", ".join(spec.alpb for spec in load_supported_solvents())
    raise ValueError(
        f"Unknown or unsupported solvent {name!r}. "
        f"Allowed solvents (ALPB names): {allowed}"
    )


def normalize_solvent(name: str) -> str:
    """Return the canonical ALPB solvent name if supported by ALPB and CPCM-X."""
    return resolve_solvent(name).alpb


def load_allowed_solvents() -> tuple[str, ...]:
    """Return canonical ALPB names for solvents supported by both models."""
    return tuple(spec.alpb for spec in load_supported_solvents())
