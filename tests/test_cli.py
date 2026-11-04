"""Comprehensive unit tests for every PEACE CLI argument and validation path."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem

import peace.main
from peace.calculators.settings import CalculatorSettings
from peace.logging_utils import (
    LogLevel,
    clear_user_warnings,
    get_log_level,
    record_user_warning,
    set_crash_on_warning,
    set_log_level,
)
from peace.main import (
    _build_calculator_settings,
    _build_cli_parser,
    _validate_cli_args,
)


def _run_peace_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["peace.main", *argv])
    runpy.run_path(str(Path(peace.main.__file__)), run_name="__main__")


# Track every CLI option tested across this module to guarantee 100% flag coverage.
EXERCISED_CLI_FLAGS: set[str] = set()


def _parse_and_validate(argv: list[str]):
    parser = _build_cli_parser()
    for token in argv:
        if token.startswith("--"):
            EXERCISED_CLI_FLAGS.add(token.split("=", 1)[0])
    args = parser.parse_args(argv)
    _validate_cli_args(parser, args)
    return parser, args


def test_cli_help_subprocess():
    """Verify `python -m peace.main --help` exits 0 and prints usage."""
    EXERCISED_CLI_FLAGS.add("--help")
    proc = subprocess.run(
        [sys.executable, "-m", "peace.main", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--smiles" in proc.stdout
    assert "--solvation" in proc.stdout
    assert "--pka" in proc.stdout


def test_cli_all_arguments_parsing_and_settings(tmp_path: Path):
    """Test parsing and settings construction for every CLI flag and calculator option."""
    ext_xyz = tmp_path / "geom.xyz"
    ext_xyz.write_text("1\ncomment\nC 0.0 0.0 0.0\n", encoding="utf-8")

    argv = [
        "--smiles",
        "NCC(=O)O",
        "--charge-min",
        "-1",
        "--charge-max",
        "1",
        "--solvation",
        "--dry-run",
        "--scratch-root",
        str(tmp_path / "scratch"),
        "--keep-scratch",
        "--keep-logs",
        "--no-plot",
        "--plot",
        "cutoff",
        "--plot-filter",
        "0.05",
        "--output-plots",
        str(tmp_path / "plots.png"),
        "--conformer-mode",
        "external_xyz",
        "--external-xyz",
        str(ext_xyz),
        "--override-solvation",
        "--output-csv",
        str(tmp_path / "out.csv"),
        "--opt-level",
        "tight",
        "--gas-calculator",
        "orca",
        "--freq-calculator",
        "orca",
        "--solv-opt-calculator",
        "orca",
        "--solv-sp-calculator",
        "orca",
        "--no-gas-optimize",
        "--xtb-version",
        "legacy",
        "--xtb-executable",
        "xtb",
        "--orca-executable",
        "orca",
        "--orca-keywords",
        "r2SCAN-3c",
        "--gas-orca-keywords",
        "wB97X-D4 def2-TZVP",
        "--freq-orca-keywords",
        "r2SCAN-3c",
        "--solv-opt-orca-keywords",
        "r2SCAN-3c",
        "--solv-sp-orca-keywords",
        "BP86 def2-TZVPD",
        "--solv-opt-orca-model",
        "smd",
        "--solv-sp-orca-model",
        "cosmors",
        "--orca-solvent",
        "water",
        "--orca-nprocs",
        "4",
        "--orca-maxcore",
        "2000",
        "--orca-extra-input",
        "%scf MaxIter 500 end",
        "--temperature",
        "310.0",
        "--solvent",
        "water",
        "--recompute-solvation",
        "--recompute-frequencies",
        "--screen-threshold",
        "20.0",
        "--max-conformers",
        "3",
        "--embedded-conformers",
        "10",
        "--conformer-energy-threshold",
        "15.0",
        "--exclude-unconverged",
        "--log-level",
        "debug",
        "--crash-on-warning",
        "--site-search-mode",
        "strong",
        "--only-protomer-search",
        "--add-tautomers",
        "[NH3+]CC(=O)[O-]",
        "--max-seed-rounds",
        "2",
        "--seed-round-cap-factor",
        "3",
        "--charge-seed-first-only",
        "--pka",
        "--proton-energy",
        "-265.9",
        "--pka-filter-type",
        "threshold",
        "--pka-filter-value",
        "5.0",
        "--output-pka-csv",
        str(tmp_path / "out_pka.csv"),
        "--output-pka-plots",
        str(tmp_path / "out_pka.png"),
    ]
    _, args = _parse_and_validate(argv)

    # Also test the --no-gxtb-optimize alias and --plot-from-csv flag
    _, alias_args = _parse_and_validate(["--no-gxtb-optimize"])
    assert alias_args.no_gas_optimize is True

    _, plot_csv_args = _parse_and_validate(["--plot-from-csv", str(tmp_path / "existing.csv")])
    assert plot_csv_args.plot_from_csv == str(tmp_path / "existing.csv")

    calc_settings: CalculatorSettings = _build_calculator_settings(args)
    calc_settings.validate()
    assert calc_settings.gas == "orca"
    assert calc_settings.freq == "orca"
    assert calc_settings.solvent_opt == "orca"
    assert calc_settings.solvation == "orca"
    assert calc_settings.uses_orca is True
    assert calc_settings.uses_xtb is False
    assert calc_settings.xtb_version == "legacy"
    assert calc_settings.orca.nprocs == 4
    assert calc_settings.orca.maxcore_mb == 2000
    assert calc_settings.orca.extra_input == "%scf MaxIter 500 end"
    assert calc_settings.orca.solvent == "water"
    assert calc_settings.resolved_orca_keywords("gas") == "wB97X-D4 def2-TZVP"
    assert calc_settings.resolved_orca_keywords("freq") == "r2SCAN-3c"
    assert calc_settings.resolved_orca_keywords("solvent_opt") == "r2SCAN-3c"
    assert calc_settings.solvation_orca_job_keywords() == "BP86 def2-TZVPD"
    assert calc_settings.solvent_opt_orca_model == "smd"
    assert calc_settings.solvation_orca_model == "cosmors"


def test_cli_validation_error_branches():
    """Verify _validate_cli_args rejects invalid flag combinations."""
    # --charge-min > --charge-max
    with pytest.raises(SystemExit):
        _parse_and_validate(["--charge-min", "2", "--charge-max", "0"])

    # --temperature <= 0
    with pytest.raises(SystemExit):
        _parse_and_validate(["--temperature", "0"])

    # --plot cutoff/count without --plot-filter
    with pytest.raises(SystemExit):
        _parse_and_validate(["--plot", "cutoff"])
    with pytest.raises(SystemExit):
        _parse_and_validate(["--plot", "count"])

    # --plot-from-csv with --no-plot
    with pytest.raises(SystemExit):
        _parse_and_validate(["--plot-from-csv", "some.csv", "--no-plot"])

    # --pka with --plot-from-csv
    with pytest.raises(SystemExit):
        _parse_and_validate(["--pka", "--solvation", "--plot-from-csv", "some.csv"])

    # --pka without --solvation
    with pytest.raises(SystemExit):
        _parse_and_validate(["--pka", "--charge-min", "0", "--charge-max", "1"])

    # --pka with single charge state
    with pytest.raises(SystemExit):
        _parse_and_validate(["--pka", "--solvation", "--charge-min", "0", "--charge-max", "0"])

    # --pka with solvent that has no tabulated proton energy and no --proton-energy
    with pytest.raises(SystemExit):
        _parse_and_validate(
            [
                "--pka",
                "--solvation",
                "--charge-min",
                "0",
                "--charge-max",
                "1",
                "--solvent",
                "acetonitrile",
            ]
        )

    # ORCA stage without keywords
    with pytest.raises(SystemExit):
        _parse_and_validate(["--gas-calculator", "orca"])

    # Invalid ORCA nprocs / maxcore
    with pytest.raises(SystemExit):
        _parse_and_validate(["--orca-nprocs", "0"])
    with pytest.raises(SystemExit):
        _parse_and_validate(["--orca-maxcore", "0"])


def test_cli_enumeration_and_site_search_flags_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Execute CLI with site-search, tautomer-addition, seed-round, and charge-seeding flags."""
    # 1. --only-protomer-search + --site-search-mode none + --add-tautomers
    out_csv_1 = tmp_path / "only_prot_none.csv"
    _run_peace_cli(
        monkeypatch,
        [
            "--smiles",
            "NCC(=O)O",
            "--only-protomer-search",
            "--site-search-mode",
            "none",
            "--add-tautomers",
            "[NH3+]CC(=O)[O-]",
            "--add-tautomers",
            "NCC(=O)O",  # duplicate, should be skipped gracefully
            "--no-plot",
            "--output-csv",
            str(out_csv_1),
        ],
    )
    df1 = pd.read_csv(out_csv_1)
    assert set(df1["protomer_smiles"]) == {"NCC(=O)O", "[NH3+]CC(=O)[O-]"}

    # 2. --max-seed-rounds 0 (keeps base protomer only)
    out_csv_2 = tmp_path / "max_rounds_0.csv"
    _run_peace_cli(
        monkeypatch,
        [
            "--smiles",
            "NCC(=O)O",
            "--max-seed-rounds",
            "0",
            "--no-plot",
            "--output-csv",
            str(out_csv_2),
        ],
    )
    df2 = pd.read_csv(out_csv_2)
    assert list(df2["protomer_smiles"]) == ["NCC(=O)O"]

    # 3. --max-seed-rounds -1, --seed-round-cap-factor, --charge-seed-first-only, --log-level verbose
    out_csv_3 = tmp_path / "charge_seed_first.csv"
    _run_peace_cli(
        monkeypatch,
        [
            "--smiles",
            "NCC(=O)O",
            "--charge-min",
            "-1",
            "--charge-max",
            "1",
            "--max-seed-rounds",
            "-1",
            "--seed-round-cap-factor",
            "2",
            "--charge-seed-first-only",
            "--site-search-mode",
            "all",
            "--log-level",
            "verbose",
            "--no-plot",
            "--output-csv",
            str(out_csv_3),
        ],
    )
    set_log_level(LogLevel.DEFAULT)
    df3 = pd.read_csv(out_csv_3)
    assert set(df3["formal_charge"]) == {-1, 0, 1}


def test_cli_plotting_and_plot_from_csv_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test --output-plots, --plot (default/cutoff/count), --plot-filter, and --plot-from-csv."""
    monkeypatch.setattr(Image.Image, "show", lambda self, *a, **kw: None)

    out_csv = tmp_path / "plot_source.csv"
    out_plot = tmp_path / "enum_plot.png"

    # Run enumeration and save plots for multiple charge states
    _run_peace_cli(
        monkeypatch,
        [
            "--smiles",
            "NCC(=O)O",
            "--charge-min",
            "0",
            "--charge-max",
            "1",
            "--output-csv",
            str(out_csv),
            "--output-plots",
            str(out_plot),
        ],
    )
    assert (tmp_path / "enum_plot_charge0.png").is_file()
    assert (tmp_path / "enum_plot_charge1.png").is_file()

    # Populate synthetic thermo columns to test --plot-from-csv with cutoff and count filters
    df = pd.read_csv(out_csv)
    df["solution_phase_free_energy_kcal_mol"] = [-100.0 + i for i in range(len(df))]
    df["delta_g_kcal_mol"] = [float(i) for i in range(len(df))]
    df["boltzmann_fraction"] = [0.8 if i == 0 else 0.2 / max(1, len(df) - 1) for i in range(len(df))]
    df.to_csv(out_csv, index=False)

    cutoff_plot = tmp_path / "from_csv_cutoff.png"
    with pytest.raises(SystemExit) as exc_info:
        _run_peace_cli(
            monkeypatch,
            [
                "--plot-from-csv",
                str(out_csv),
                "--plot",
                "cutoff",
                "--plot-filter",
                "0.5",
                "--output-plots",
                str(cutoff_plot),
            ],
        )
    assert exc_info.value.code == 0
    assert cutoff_plot.is_file()

    count_plot = tmp_path / "from_csv_count.png"
    with pytest.raises(SystemExit) as exc_info:
        _run_peace_cli(
            monkeypatch,
            [
                "--plot-from-csv",
                str(out_csv),
                "--plot",
                "count",
                "--plot-filter",
                "1",
                "--output-plots",
                str(count_plot),
            ],
        )
    assert exc_info.value.code == 0
    assert count_plot.is_file()


def test_cli_dry_run_solvation_and_pka_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_executable_checks,
):
    """Test --solvation --dry-run with --pka, --conformer-mode external_xyz, and related flags."""
    # Write a valid 3D glycine XYZ file for external_xyz mode
    mol = Chem.AddHs(Chem.MolFromSmiles("NCC(=O)O"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    xyz_path = tmp_path / "glycine.xyz"
    xyz_path.write_text(Chem.MolToXYZBlock(mol), encoding="utf-8")

    out_csv = tmp_path / "dry_solv.csv"
    out_pka_csv = tmp_path / "dry_solv_pka.csv"
    out_pka_plot = tmp_path / "dry_solv_pka.png"
    scratch_root = tmp_path / "scratch_dry"

    _run_peace_cli(
        monkeypatch,
        [
            "--smiles",
            "NCC(=O)O",
            "--charge-min",
            "0",
            "--charge-max",
            "1",
            "--solvation",
            "--dry-run",
            "--scratch-root",
            str(scratch_root),
            "--keep-scratch",
            "--keep-logs",
            "--conformer-mode",
            "external_xyz",
            "--external-xyz",
            str(xyz_path),
            "--opt-level",
            "normal",
            "--gas-calculator",
            "aimnet2",
            "--solv-opt-calculator",
            "aimnet2",
            "--no-gas-optimize",
            "--temperature",
            "300.0",
            "--solvent",
            "acetonitrile",
            "--recompute-solvation",
            "--recompute-frequencies",
            "--screen-threshold",
            "30.0",
            "--max-conformers",
            "2",
            "--embedded-conformers",
            "10",
            "--conformer-energy-threshold",
            "20.0",
            "--exclude-unconverged",
            "--pka",
            "--proton-energy",
            "-260.0",
            "--pka-filter-type",
            "count",
            "--pka-filter-value",
            "5",
            "--output-csv",
            str(out_csv),
            "--output-pka-csv",
            str(out_pka_csv),
            "--output-pka-plots",
            str(out_pka_plot),
            "--no-plot",
        ],
    )

    assert out_csv.is_file()
    assert out_pka_csv.is_file()
    df = pd.read_csv(out_csv)
    assert set(df["formal_charge"]) == {0, 1}
    assert set(df["solvent"]) == {"acetonitrile"}


def test_crash_on_warning_and_log_level():
    """Verify --crash-on-warning aborts immediately when record_user_warning is called."""
    try:
        clear_user_warnings()
        set_log_level("debug")
        assert get_log_level() == LogLevel.DEBUG
        set_crash_on_warning(True)
        with pytest.raises(SystemExit, match="Crash-on-warning"):
            record_user_warning("test warning", context="unit_test")
    finally:
        set_crash_on_warning(False)
        set_log_level("default")
        clear_user_warnings()


def test_all_cli_flags_are_exercised():
    """Meta-test ensuring every option string registered on _build_cli_parser() was tested."""
    parser = _build_cli_parser()
    all_long_flags = {
        opt
        for action in parser._actions
        for opt in action.option_strings
        if opt.startswith("--")
    }
    missing = all_long_flags - EXERCISED_CLI_FLAGS
    assert not missing, f"CLI flags not exercised by tests: {sorted(missing)}"
