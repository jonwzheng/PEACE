# PEACE: Protomer Enumeration and Charge Engine

![PEACE Logo](docs/static/header.svg)

> [!WARNING]
> This repository is actively being developed and is not yet complete. Expect breaking changes and incomplete documentation.


More details forthcoming.

## Getting started
Simply clone this repo and add this base directory to PATH, or run calculations directly from here.

For the estimation of individual protomer populations, the user needs to install CREST, xTB with CPCM-X, and g-xTB with the respective binary or execution script added to path. Optionally, [ORCA](https://www.faccts.de/docs/orca/6.0/tutorials/index.html) can also be installed and placed on `PATH`: after the default xTB / g-xTB solvation workflow (`python -m peace.main --optimize ...`), you may add **`--refine`** to re-evaluate solvation with ORCA’s openCOSMO-RS for protomers within **`--refine-threshold`** (kcal/mol) of the lowest g-xTB solution-phase energy. This gives a somewhat more accurate evaluation of the solvation energy, which is added to the g-xTB gas-phase energies and xTB frequency calculation.

## How it works:

1. **Tautomer Enumeration** using RDKit.
2. **Protomer Enumeration** by searching each tautomer for acid/base sites and sequentially (de)protonating all possible combinations (this searches for zwitterion forms).
3. **Visualization** of all relevant tautomer-protomers based on their graph representations. 
4. **(Optional) Microstate Population Estimation** using quantum-chemical calculations. It is planned that a machine learning model will be made available to speed this up.