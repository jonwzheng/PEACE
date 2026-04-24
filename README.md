# PEACE

![PEACE Logo](docs/static/header.svg)

Protomer Enumeration and Charge Engine

More details forthcoming.

## Getting started
Simply clone this repo and add this base directory to PATH, or run calculations directly from here.

For the estimation of individual protomer populations, the user needs to install CREST, xTB, and [OPTIONAL; NOT IMPLEMENTED YET] g-xTB with the respective binary or execution script added to path. (g-xTB awaiting implicit solvation additions)

## How it works:

- Enumerates relevant tautomers for a given structure.
- For each tautomer, embed a single protomer (this is its existing SMILES-based structure)
- From the base protomer, searches for acid/base sites and sequentially deprotonates/protonates all possible enumerations. Embeds those protomers toward that tautomer. (this gives us zwitterions)
- Each of these protomer/tautomer pairs is now saved. 
- Optionally, there is a solvation energy workflow for computing the relative micropopulations.