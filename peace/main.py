from peace.protomer import Species
from peace.engine import ChargeEngine
from peace.common import show_images
from peace import __version__
from datetime import datetime
import time
from pathlib import Path

def _build_cli_parser():
    import argparse

    p = argparse.ArgumentParser(description="PEACE demo: tautomer/protomer enumeration + optional xTB solvation workflow.")
    p.add_argument("--smiles", type=str, default="NCCCC(=O)CCC(=O)O", help="Input SMILES to enumerate.")
    p.add_argument(
        "--optimize",
        action="store_true",
        help="If set, run the solvation energy workflow for all generated protomers (requires xTB + g-xTB binaries).",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not run calculations.")
    p.add_argument("--scratch-root", type=str, default="./solvation_results", help="Scratch root for xTB runs.")
    p.add_argument("--keep-scratch", action="store_true", help="Keep xTB scratch directories after each protomer run.")
    p.add_argument(
        "--keep-logs",
        action="store_true",
        help="Preserve run stdout logs into a separate log folder after each run.",
    )
    p.add_argument("--no-plot", action="store_true", help="Skip RDKit image rendering.")
    p.add_argument(
        "--conformer-mode",
        type=str,
        default="mmff94",
        choices=["mmff94", "external_xyz", "skip_search"],
        help="Conformer geometry input for xTB runs.",
    )
    p.add_argument("--external-xyz", type=str, default=None, help="Path to external xyz (used only with conformer-mode=external_xyz).")
    p.add_argument(
        "--override-solvation",
        action="store_true",
        help="Override any existing species solvation folder results.",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default="results.csv",
        help="Path to save the final dataframe as CSV.",
    )
    p.add_argument(
        "--opt-level",
        type=str,
        default="loose",
        choices=["loose", "tight", "vtight"],
        help="Optimization level for xTB runs.",

    )
    p.add_argument(
        "--parse-solvation",
        type=str,
        default="g",
        choices=["g", "e"],
        help="Solvation energy parser mode: 'g' for dG_solv (default), 'e' for Gsolv.",
    )
    p.add_argument(
        "--sp-energy",
        type=str,
        default="gxtb",
        choices=["gxtb", "xtb"],
        help="Gas-phase SP source: 'gxtb' (driver SP, default) or 'xtb' (from Hessian output).",
    )
    return p


def _make_species(smiles: str, *, engine: ChargeEngine) -> Species:
    spec = Species.from_smiles(smiles)
    tautomers = engine.search_for_tautomers(spec)
    spec.embed_tautomers_from_list_of_smiles(tautomers)

    return spec


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


def _header_banner() -> str:
    return """ RUNNING PEACE...
                         @@@%%@@@%@@@%%%@@              
                       @%@%*=:-####**#*--+#@%@@          
                    @%%+.    :#********#:   .+@@@        
                  @%*:       #***********      :#@@      
                @@%:         ************        .*@@    
               @@+           .#*******#*           -@@   
              @@-              +#####@*             .@@  
             @@-                .:--.                :@@ 
             @#                   ..                  =@ 
            @@.                .:-=-:.                 @@
            @#               .+*+====+*-               #@
            @+              :+-.......:=*              +@
            @+             -=...........:*             +@
            @#             *:............=-            #@
            @@             +.............+:           .@@
            @@+    ..     .-+...........-+.    ..:.   #@ 
             @@: .==:+-...  .*-:......=+=  ...--: =: =@@ 
              @@:-    ::     .=**++++*%:     -.    =+@@  
               @@=   :+.        ..-:.        .=   ++@@   
                @@*=-#.          .--           --+@@@    
                  %@+.          -:. =.         =%@@      
                    %%*:       -     *      .+@%@        
                      @%%#*=-:..=   *.  :=#%%@           
                          @@%%%@@%##@%%%@@@              
    """

if __name__ == "__main__":
    start_ts = time.time()
    args = _build_cli_parser().parse_args()
    run_started_at = _ts()
    _log(_header_banner())
    _log(f"Version: {__version__}")
    _log(f"Run started at: {run_started_at}")
    _log(f"Input SMILES: {args.smiles}")

    engine = ChargeEngine()
    spec = _make_species(args.smiles, engine=engine)

    # log the tautomers found
    tautomer_items = list(spec.tautomers.items())
    _log(f"Tautomer enumeration complete: {len(tautomer_items)} tautomer(s) found")
    for taut_idx, taut in tautomer_items:
        smiles = taut.protomers[0].smiles if 0 in taut.protomers else "N/A"
        _log(f"  Tautomer {taut_idx + 1}/{len(tautomer_items)}: {smiles}")

    _log("Enumerating protomeric forms for each tautomer")
    for taut_idx, taut in tautomer_items:
        acid_sites = engine.search_ionization_centers(taut, "acidic")
        basic_sites = engine.search_ionization_centers(taut, "basic")
        _log(
            f"  Tautomer {taut_idx + 1}/{len(tautomer_items)} ionization sites -> "
            f"acidic={acid_sites if acid_sites else '[]'}, basic={basic_sites if basic_sites else '[]'}"
        )
        taut.generate_protomers_from_base_protomer(acid_sites, basic_sites)
        _log(
            f"  Tautomer {taut_idx + 1}/{len(tautomer_items)} protomers found: {len(taut.protomers)}"
        )
        for prot_idx, prot in taut.protomers.items():
            _log(f"    Protomer {prot_idx + 1}/{len(taut.protomers)}: {prot.smiles}")

    if args.optimize:
        from peace.solvation import run_protomer_solvation
        import shutil

        scratch_root_path = Path(args.scratch_root)
        species_scratch = scratch_root_path / f"species_{spec.key}"
        if species_scratch.exists():
            if args.override_solvation:
                _log(f"Override enabled: removing existing optimization folder {species_scratch}")
                shutil.rmtree(species_scratch, ignore_errors=True)
            else:
                raise FileExistsError(
                    "Existing solvation results detected. "
                    f"Found existing species folder: {species_scratch}. "
                    "Use --override-solvation to delete prior results and rerun."
                )
        species_scratch.mkdir(parents=True, exist_ok=True)

        _log("Optimization requested: starting per-protomer workflow")
        for taut_idx, taut in tautomer_items:
            protomer_items = list(taut.protomers.items())
            for prot_idx, protomer in protomer_items:
                prefix = (
                    f"tautomer {taut_idx + 1}/{len(tautomer_items)} "
                    f"protomer {prot_idx + 1}/{len(protomer_items)}"
                )
                _log(f"Optimizing {prefix}")
                run_protomer_solvation(
                    protomer,
                    protomer_id=str(prot_idx),
                    scratch_root=species_scratch / f"tautomer_{taut_idx}",
                    conformer_mode=args.conformer_mode,
                    external_xyz_path=args.external_xyz,
                    keep_scratch=bool(args.keep_scratch),
                    keep_logs=bool(args.keep_logs),
                    parse_solvation=args.parse_solvation,
                    sp_energy=args.sp_energy,
                    dry_run=bool(args.dry_run),
                    opt_level=args.opt_level,
                    progress_callback=lambda stage, prefix=prefix: _log(f"  [{prefix}] {stage}"),
                )
        _log(f"Optimization outputs saved under: {species_scratch}")

        _log("Calculating Boltzmann populations")
        spec.assign_boltzmann_microstate_populations(temperature_k=298.15)
        f_zwit = spec.get_f_zwit()
        _log(f"Total predicted zwitterion fraction (f_zwit): {f_zwit:.5f}")
        for taut_idx, taut in tautomer_items:
            _log(f"  Tautomer {taut_idx + 1}/{len(tautomer_items)} Boltzmann populations:")
            for prot_idx, protomer in taut.protomers.items():
                frac = (
                    protomer.mol.GetProp("peace_boltzmann_fraction")
                    if protomer.mol is not None and protomer.mol.HasProp("peace_boltzmann_fraction")
                    else "N/A"
                )
                _log(f"    Protomer {prot_idx + 1}/{len(taut.protomers)} ({protomer.smiles}): {frac}")
    if not args.no_plot:
        imgs = spec.generate_protomer_plot(n_columns=5)
        show_images(imgs, mode="vertical")
    df = spec.to_dataframe()
    print(df)

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        _log(f"Saved dataframe CSV to: {output_path.resolve()}")

    end_ts = time.time()
    _log(f"Run finished at: {_ts()}")
    _log(f"Execution time: {end_ts - start_ts:.2f} s")
