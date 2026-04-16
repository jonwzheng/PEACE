# PEACE

![PEACE Logo](docs/static/header.svg)

Protomer Enumeration and Charge Engine

More details forthcoming.


How it works:

- Enumerates relevant tautomers for a given structure.
- For each tautomer, embed a single protomer (this is its existing SMILES-based structure)
- From the base protomer, searches for acid/base sites and sequentially deprotonates/protonates all possible enumerations. Embeds those protomers toward that tautomer. (this gives us zwitterions)
- Each of these protomer/tautomer pairs is now saved. 

Solvated energy quick calculation workflow:

1. g-xTB optimization with implicit solvent and loose coordinates
```xtb test.xyz --driver "gxtb -grad -c xtbdriver.xyz" --opt loose --alpb water```

2. CPCM-X calculation
(TODO)

3. Frequency calculation
(TODO)