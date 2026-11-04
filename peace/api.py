"""
High-level Python API mirroring the PEACE command-line interface.

Every CLI flag is available as a keyword argument named after its argparse
destination, i.e. the flag with the leading dashes removed and remaining dashes
replaced by underscores (``--charge-min`` -> ``charge_min``,
``--solv-sp-calculator`` -> ``solv_sp_calculator``). Use ``cli_defaults()`` to
list them.

Example::

    import peace

    result = peace.solvate("NCC(=O)O", max_conformers=2)
    print(result.dataframe[["protomer_smiles", "boltzmann_fraction"]])
"""

from __future__ import annotations

from os import PathLike
from typing import Any

from peace.main import PeaceResult, _build_cli_parser, run_from_args

# Defaults that differ from the CLI: the API does not write files or open
# image windows unless asked to.
_API_DEFAULT_OVERRIDES: dict[str, Any] = {
    "no_plot": True,
    "output_csv": None,
}


def _raise_value_error(message: str) -> None:
    raise ValueError(message)


def _build_api_parser():
    parser = _build_cli_parser()
    parser.error = _raise_value_error
    return parser


def cli_defaults() -> dict[str, Any]:
    """Return every accepted keyword argument with its default value."""
    defaults = vars(_build_cli_parser().parse_args([]))
    defaults.update(_API_DEFAULT_OVERRIDES)
    return defaults


def _normalize_option(key: str, value: Any) -> Any:
    if isinstance(value, PathLike):
        return str(value)
    if key == "add_tautomers" and isinstance(value, str):
        return [value]
    if key == "add_tautomers" and value is not None:
        return [str(v) for v in value]
    return value


def run(smiles: str, **options: Any) -> PeaceResult:
    """
    Run PEACE on ``smiles`` with the same options as the CLI.

    Raises ``TypeError`` for unknown options and ``ValueError`` for invalid
    option combinations (the cases where the CLI would exit with an error).
    """
    parser = _build_api_parser()
    args = parser.parse_args([])
    for key, value in _API_DEFAULT_OVERRIDES.items():
        setattr(args, key, value)

    valid_keys = set(vars(args))
    unknown = sorted(set(options) - valid_keys)
    if unknown:
        raise TypeError(
            f"Unknown PEACE option(s): {', '.join(unknown)}. "
            "See peace.cli_defaults() for accepted keyword arguments."
        )

    args.smiles = str(smiles)
    for key, value in options.items():
        setattr(args, key, _normalize_option(key, value))

    return run_from_args(parser, args, print_results=False)


def enumerate_protomers(
    smiles: str,
    *,
    charge_min: int = 0,
    charge_max: int = 0,
    **options: Any,
) -> PeaceResult:
    """Enumerate tautomers and protomers over ``[charge_min, charge_max]`` without QM calculations."""
    if options.get("solvation"):
        raise ValueError("enumerate_protomers() does not run solvation; use solvate() instead.")
    options.pop("solvation", None)
    return run(smiles, charge_min=charge_min, charge_max=charge_max, solvation=False, **options)


def solvate(
    smiles: str,
    *,
    charge_min: int = 0,
    charge_max: int = 0,
    **options: Any,
) -> PeaceResult:
    """Enumerate tautomers/protomers and compute solution-phase free energies and populations."""
    options.pop("solvation", None)
    return run(smiles, charge_min=charge_min, charge_max=charge_max, solvation=True, **options)
