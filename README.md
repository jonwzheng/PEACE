# PEACE: Protomer Enumerator & Automatic Calculator for Energies

![PEACE Logo](https://raw.githubusercontent.com/jonwzheng/PEACE/main/docs/static/header.svg)

## Overview
PEACE is a lightweight cheminformatics workflow for enumerating acid-base tautomers. It also includes a calculator function that orchestrates QM calculations for estimating the microstate populations in numerous solvent systems. 

For more information, please read the manuscript (and cite, if using this work):
> Zheng, J. W., & Heid, E. (2026). An Atomistic Workflow for Determining Acid-Base Prototropic Tautomer Microstate Populations. RSC Advances.

## Installation
1. Install from PyPI with `pip install peace-protomer`.
2. If using free energy simulations, additionally install the appropriate QM software:
  - If using `xtb`-based methods, install `gxtb` version 2.0.1 or greater into `bin` (or specify its location with `--xtb-executable`). 
  - If using DFT, install ORCA 6.x and add this to `bin` (or specify its location with `--orca-executable`).
  - If using AIMNet2, install it into the same Python environment as PEACE with `pip install "aimnet[ase]"`.

As an alternative to step 1, you can install from source:
```
git clone git@github.com:jonwzheng/PEACE.git
cd PEACE
python -m venv peace_env
source peace_env/bin/activate
pip install -e ".[test]"
pytest
```

Example run:
`peace --smiles "NCC(=O)O" --solvation` (equivalently `python -m peace.main ...`)

CLI commands can be displayed e.g. with:
`peace --help`

## Python API
Everything available on the CLI can also be run from Python e.g. (`--charge-min` -> `charge_min`, `--solv-sp-calculator` -> `solv_sp_calculator`). You can check their defaults with `peace.cli_defaults()`. By default, the Python API does not write a CSV or display plots unless you pass `output_csv=...` or `no_plot=False`.

```python
import peace

# Tautomer/protomer enumeration only (no QM)
result = peace.enumerate_protomers("O=C(O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O", charge_min=-4, charge_max=2)
for charge, species in result.species_by_charge.items():
    print(charge, sum(len(t.protomers) for t in species.tautomers.values()))

# Solvation free energies and Boltzmann populations
result = peace.solvate("NCC(=O)O", max_conformers=2, scratch_root="./solvation_results")
print(result.dataframe[["protomer_smiles", "boltzmann_fraction"]])

# Any CLI option combination, e.g. pKa
result = peace.solvate("NCC(=O)O", charge_min=0, charge_max=1, pka=True)
print(result.pka_dataframe)
```

## How it works:

1. **Tautomer Enumeration** using RDKit.
2. **Protomer Enumeration** by searching each tautomer for acid/base sites and sequentially (de)protonating all possible combinations (this searches for zwitterion forms).
3. **(Optional) Microstate Population Estimation** using quantum-chemical calculations, G = E_gas + G_RRHO + dG_solv:
   - **Screening**: KDG conformer -> implicit-solvation SP, gas-phase SP, and frequencies (RRHO) on the screening geometry.
   - **Refinement** (screened-in protomers): MMFF94-ranked conformer ensemble -> implicit-solvent optimization -> implicit-solvation SP on that geometry -> gas-phase re-optimization and gas-phase SP -> frequencies on the gas-phase geometry. Pass `--no-gas-optimize` to skip the gas-phase re-optimization and evaluate the gas SP and frequencies at the implicit-solvent geometry.
   It is planned that a machine learning model will be made available to speed this up.
4. **Visualization** of all relevant tautomer-protomers based on their graph representations. 

## Calculators
Each stage of the free-energy workflow has its own calculator:

| Stage | Flag | Default | Options |
| --- | --- | --- | --- |
| Gas-phase SP + gas-phase re-optimization | `--gas-calculator` | `gxtb` | `gxtb`, `gfn2`, `aimnet2`, `orca` |
| Frequencies (RRHO) | `--freq-calculator` | `gfn2` | `gfn2`, `gxtb`, `orca` |
| Implicit-solvent optimization | `--solv-opt-calculator` | `gfn2` (GFN2-xTB/ALPB) | `gfn2`, `aimnet2`, `orca` |
| Implicit-solvation free energy | `--solv-sp-calculator` | `cpcmx` (CPCM-X) | `cpcmx`, `orca` |

`--freq-calculator gxtb` computes the RRHO contribution with g-xTB instead of GFN2-xTB frequencies; it may be slightly more accurate but is about 2x slower.

ORCA stages take arbitrary method keywords: `--orca-keywords` sets a default for every ORCA stage, and `--gas-orca-keywords`, `--freq-orca-keywords`, `--solv-opt-orca-keywords`, `--solv-sp-orca-keywords` override it per stage. PEACE adds the job type (`Opt`/`FREQ`) and the solvation model itself: `--solv-opt-orca-model` (`cpcm`, `smd`) and `--solv-sp-orca-model` (`cpcm`, `smd`, `cosmors`). See `python -m peace.main --help` for more runtime options.

Example: r2SCAN-3c gas-phase energies and frequencies, keeping GFN2-xTB/ALPB optimization and CPCM-X solvation:
```
python -m peace.main --smiles "NCC(=O)O" --solvation \
    --gas-calculator orca --freq-calculator orca --orca-keywords "r2SCAN-3c"
```

## pKa Prediction
The free energies of tautomers, and their ensembles, can be used to estimate pKa values. It is currently quite inaccurate and should be considered work in progress, or at least a way of estimating which micro-reactions contribute to acidity. Work is ongoing.

You can call this with the `--pka` flag as well as `--proton-energy` to estimate the proton free energy in solution.