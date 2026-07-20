# PEACE: Protomer Enumeration and Computed Energies

![PEACE Logo](docs/static/header.svg)

> [!WARNING]
> This repository is actively being developed and is not yet complete. Expect breaking changes and incomplete documentation.


More details forthcoming.

## Getting started
Simply clone this repo and add this base directory to PATH, or run calculations directly from here.

For the estimation of individual protomer populations, the user needs to install CREST, xTB with CPCM-X, and g-xTB with the respective binary or execution script added to path.

## How it works:

1. **Tautomer Enumeration** using RDKit.
2. **Protomer Enumeration** by searching each tautomer for acid/base sites and sequentially (de)protonating all possible combinations (this searches for zwitterion forms).
3. **Visualization** of all relevant tautomer-protomers based on their graph representations. 
4. **(Optional) Microstate Population Estimation** using quantum-chemical calculations:
   - **Screening**: KDG conformer → CPCM-X, g-xTB gas-phase SP, and RRHO on the screening geometry.
   - **Refinement** (screened-in protomers): MMFF94-ranked conformer ensemble → GFN2-xTB/ALPB optimization → CPCM-X, g-xTB SP, and RRHO on the ALPB geometry by default.
   - Pass `--gxtb-optimize` to additionally re-optimize each refined conformer at g-xTB gas phase before its g-xTB SP and frequency steps.
   It is planned that a machine learning model will be made available to speed this up.