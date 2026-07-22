# PEACE: Protomer Enumerator & Automatic Calculator for Energies

![PEACE Logo](docs/static/header.svg)

> [!WARNING]
> This repository is actively being developed. Expect breaking changes and incomplete documentation.


## Getting started
1. Clone this repo and add this base directory to PATH, or run calculations directly from here.
2. Install the project dependencies in `pyproject.toml` using `uv` or `pip`.
3. If using free energy simulations, install `gxtb` version 2.0.1 or greater into `bin` (or specify its location with `--xtb-executable`). If running into issues, one can also install `gxtb` version 2.0.0, and separately install `xtb` and `cpcm` because the solvation model was not included in the 2.0.0 binary. (If this is done, set `--xtb-version` to `legacy`.)

Example setup with pip:
```
python -m venv peace_env
source peace_env/bin/activate
pip install -e .
```

Example run:
`python -m peace.main --smiles "NCC(=O)O" --solvation`

CLI commands can be displayed e.g. with:
`python -m peace.main --help`

## How it works:

1. **Tautomer Enumeration** using RDKit.
2. **Protomer Enumeration** by searching each tautomer for acid/base sites and sequentially (de)protonating all possible combinations (this searches for zwitterion forms).
3. **(Optional) Microstate Population Estimation** using quantum-chemical calculations:
   - **Screening**: KDG conformer → CPCM-X, g-xTB gas-phase SP, and RRHO on the screening geometry.
   - **Refinement** (screened-in protomers): MMFF94-ranked conformer ensemble -> GFN2-xTB/ALPB optimization -> re-optimize at g-xTB gas phase.CPCM-X on GFN2-xTB/ALPB geometry; g-xTB SP and RRHO on the g-xTB geometry by default. Pass `--no-gxtb-optimize` to instead have all energy calculations done at the GFN2-xTB/ALPB geometry. 
   It is planned that a machine learning model will be made available to speed this up.
4. **Visualization** of all relevant tautomer-protomers based on their graph representations. 