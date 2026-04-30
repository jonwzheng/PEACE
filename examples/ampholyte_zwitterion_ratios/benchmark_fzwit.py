import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "molecule"


def _compute_predicted_f_zwit(results_csv: Path) -> float:
    df = pd.read_csv(results_csv)
    if "is_zwitterion" not in df.columns or "boltzmann_fraction" not in df.columns:
        raise ValueError(
            f"Expected columns 'is_zwitterion' and 'boltzmann_fraction' in {results_csv}."
        )

    zwit = df["is_zwitterion"].astype(str).str.lower().isin(["true", "1", "yes"])
    fractions = pd.to_numeric(df["boltzmann_fraction"], errors="coerce").fillna(0.0)
    return float(fractions[zwit].sum())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark predicted zwitterion fractions against tautomer_ratios_test.csv."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/tautomer_ratios_test.csv",
        help="Input CSV with SMILES and experimental f_zwit.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/f_zwit_benchmark",
        help="Folder to store per-molecule PEACE outputs and aggregate benchmark CSV.",
    )
    return parser


def main() -> None:
    args, main_extra_args = _build_parser().parse_known_args()
    repo_root = Path(__file__).resolve().parents[2]

    input_csv = (repo_root / args.input_csv).resolve()
    results_root = (repo_root / args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_csv)
    if "SMILES" not in data.columns or "f_zwit" not in data.columns:
        raise ValueError("Input CSV must contain 'SMILES' and 'f_zwit' columns.")

    rows: list[dict] = []
    for idx, row in data.iterrows():
        name = str(row.get("molecule", f"row_{idx}"))
        smiles = str(row["SMILES"])
        experimental = float(row["f_zwit"])
        mol_dir = results_root / f"{idx:03d}_{_slugify(name)}"
        mol_dir.mkdir(parents=True, exist_ok=True)

        output_csv = mol_dir / "results.csv"
        scratch_root = mol_dir / "solvation"

        cmd = [
            sys.executable,
            "-m",
            "peace.main",
            "--smiles",
            smiles,
            "--optimize",
            "--override-solvation",
            "--scratch-root",
            str(scratch_root),
            "--output-csv",
            str(output_csv),
            "--no-plot",
        ]
        cmd.extend(main_extra_args)

        print(f"[{idx + 1}/{len(data)}] Running: {name}")
        run = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        (mol_dir / "stdout.log").write_text(run.stdout)
        (mol_dir / "stderr.log").write_text(run.stderr)

        predicted = None
        abs_error = None
        if run.returncode == 0 and output_csv.exists():
            predicted = _compute_predicted_f_zwit(output_csv)
            abs_error = abs(predicted - experimental)

        rows.append(
            {
                "molecule": name,
                "SMILES": smiles,
                "experimental_f_zwit": experimental,
                "predicted_f_zwit": predicted,
                "abs_error": abs_error,
                "run_ok": run.returncode == 0,
                "run_returncode": run.returncode,
                "result_dir": str(mol_dir),
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = results_root / "benchmark_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved benchmark summary to: {out_path}")


if __name__ == "__main__":
    main()
