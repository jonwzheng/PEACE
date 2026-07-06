"""Solvent validation for combined ALPB optimization and CPCM-X solvation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ALPB_SOLVENTS_FILE = _PACKAGE_ROOT / "data" / "allowed_solvents.txt"
CPCM_SOLVENTS_FILE = _PACKAGE_ROOT / "data" / "cpcm_allowed_solvents.txt"
CPCM_INTERNAL_SOLVENTS_FILE = _PACKAGE_ROOT / "data" / "cpcm_internal_solvent_names.txt"

# ALPB names that must not be exposed even when a CPCM alias exists.
_BLOCKED_ALPB_SOLVENTS = frozenset({"ch2cl2", "chcl3", "ether", "woctanol", "hexandecane"})

# ALPB solvents whose internal CPCM-X flag name differs from all aliases on their
# cpcm_allowed_solvents.txt entry.
_ALPB_INTERNAL_CPCM_OVERRIDES: dict[str, str] = {
    "methanol": "methoxyethanol",
}


@dataclass(frozen=True)
class SolventNames:
    """Canonical ALPB and internal CPCM-X solvent names for one supported phase."""

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
        if not name or name.startswith("#"):
            continue
        if name.endswith(".py"):
            continue
        solvents.append(name)
    if not solvents:
        raise ValueError(f"No solvents listed in {path}")
    return tuple(solvents)


@lru_cache(maxsize=1)
def load_alpb_solvents(path: Path | None = None) -> tuple[str, ...]:
    return _parse_solvent_list(path or ALPB_SOLVENTS_FILE)


@lru_cache(maxsize=1)
def load_cpcm_internal_solvent_names(path: Path | None = None) -> frozenset[str]:
    return frozenset(_parse_solvent_list(path or CPCM_INTERNAL_SOLVENTS_FILE))


CpcmSolventEntry = tuple[tuple[str, ...], frozenset[str]]


@lru_cache(maxsize=1)
def load_cpcm_solvent_entries(
    path: Path | None = None,
) -> tuple[CpcmSolventEntry, ...]:
    solvent_file = path or CPCM_SOLVENTS_FILE
    if not solvent_file.is_file():
        raise FileNotFoundError(f"CPCM solvent list not found: {solvent_file}")

    entries: list[tuple[tuple[str, ...], frozenset[str]]] = []
    for line in solvent_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        aliases = tuple(part.strip() for part in re.split(r"\s*/\s*", stripped) if part.strip())
        if not aliases:
            continue
        keys = frozenset(_normalize_key(alias) for alias in aliases)
        entries.append((aliases, keys))
    if not entries:
        raise ValueError(f"No CPCM solvents listed in {solvent_file}")
    return tuple(entries)


def _internal_name_lookup(internal_names: frozenset[str]) -> dict[str, str]:
    return {_normalize_key(name): name for name in internal_names}


def _internal_cpcm_name_for_alpb(
    alpb_name: str,
    cpcm_entries: tuple[CpcmSolventEntry, ...],
    internal_lookup: dict[str, str],
) -> str | None:
    override = _ALPB_INTERNAL_CPCM_OVERRIDES.get(alpb_name)
    if override is not None:
        override_key = _normalize_key(override)
        if override_key in internal_lookup:
            return internal_lookup[override_key]
        return None

    alpb_key = _normalize_key(alpb_name)
    for aliases, alias_keys in cpcm_entries:
        if alpb_key not in alias_keys:
            continue

        if alpb_key in internal_lookup:
            return internal_lookup[alpb_key]

        for alias in aliases:
            alias_key = _normalize_key(alias)
            if alias_key in internal_lookup:
                return internal_lookup[alias_key]
        return None
    return None


@lru_cache(maxsize=1)
def load_supported_solvents() -> tuple[SolventNames, ...]:
    """Return solvents supported by ALPB, CPCM-X aliases, and internal CPCM names."""
    cpcm_entries = load_cpcm_solvent_entries()
    internal_lookup = _internal_name_lookup(load_cpcm_internal_solvent_names())
    supported: list[SolventNames] = []
    missing_cpcm: list[str] = []
    missing_internal: list[str] = []

    for alpb_name in load_alpb_solvents():
        if alpb_name in _BLOCKED_ALPB_SOLVENTS:
            continue

        alpb_key = _normalize_key(alpb_name)
        matched_entry = next(
            (aliases for aliases, alias_keys in cpcm_entries if alpb_key in alias_keys),
            None,
        )
        if matched_entry is None:
            missing_cpcm.append(alpb_name)
            continue

        internal_name = _internal_cpcm_name_for_alpb(alpb_name, cpcm_entries, internal_lookup)
        if internal_name is None:
            missing_internal.append(alpb_name)
            continue
        supported.append(SolventNames(alpb=alpb_name, cpcm=internal_name))

    if missing_cpcm:
        raise ValueError(
            "ALPB solvents without a CPCM-X alias mapping: "
            + ", ".join(sorted(missing_cpcm))
        )
    if missing_internal:
        raise ValueError(
            "ALPB solvents without an internal CPCM-X name: "
            + ", ".join(sorted(missing_internal))
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
        alpb_key = _normalize_key(spec.alpb)
        for aliases, alias_keys in cpcm_entries:
            if alpb_key not in alias_keys:
                continue
            for alias in aliases:
                lookup.setdefault(_normalize_key(alias), spec)
            break
    return lookup


def resolve_solvent(name: str) -> SolventNames:
    """Resolve a user solvent name to canonical ALPB and internal CPCM-X names."""
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
