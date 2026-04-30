from peace.protomer import Species
from peace.engine import ChargeEngine
from peace.common import show_images
from peace import __version__
from datetime import datetime
import time
from pathlib import Path
from typing import Any, Optional

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
    p.add_argument(
        "--screen-threshold",
        type=float,
        default=15.0,
        help="Exclude protomers from full post-screen optimization if screening delta exceeds energy threshold (kcal/mol).",
    )
    p.add_argument(
        "--exclude-unconverged",
        action="store_true",
        help="Exclude connectivity-mismatch protomers from Boltzmann weighting and f_zwit while still reporting them.",
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


def _set_optional_double_prop(protomer, key: str, value: Optional[float]) -> None:
    if protomer.mol is None or value is None:
        return
    protomer.mol.SetDoubleProp(key, float(value))


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
        from peace.solvation import run_protomer_solvation, run_protomer_screening
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

        _log(" *** SCREENING PROTOMERS *** ")
        screening_records: list[tuple[int, int, Any, Optional[float]]] = []
        for taut_idx, taut in tautomer_items:
            protomer_items = list(taut.protomers.items())
            for prot_idx, protomer in protomer_items:
                prefix = (
                    f"tautomer {taut_idx + 1}/{len(tautomer_items)} "
                    f"protomer {prot_idx + 1}/{len(protomer_items)}"
                )
                _log(f"Screening {prefix}")
                screening_result = run_protomer_screening(
                    protomer,
                    protomer_id=f"{prot_idx}_screen",
                    scratch_root=species_scratch / f"tautomer_{taut_idx}" / "screening",
                    conformer_mode=args.conformer_mode,
                    external_xyz_path=args.external_xyz,
                    opt_level=args.opt_level,
                    keep_scratch=bool(args.keep_scratch),
                    keep_logs=bool(args.keep_logs),
                    parse_solvation=args.parse_solvation,
                    dry_run=bool(args.dry_run),
                    progress_callback=lambda stage, prefix=prefix: _log(f"  [{prefix}] {stage}"),
                )
                screening_records.append(
                    (
                        taut_idx,
                        prot_idx,
                        protomer,
                        screening_result.solution_phase_free_energy_kcal_mol,
                    )
                )

        valid_screening = [row for row in screening_records if row[3] is not None]
        min_screening_solution_energy = min((row[3] for row in valid_screening), default=None)
        if min_screening_solution_energy is None:
            _log("Screening did not produce any valid solution-phase energies; keeping all protomers for full optimization.")

        protomers_to_optimize: list[tuple[int, int, Any, Optional[float], Optional[float]]] = []
        screened_out: list[tuple[int, int, Any, Optional[float], Optional[float]]] = []
        for taut_idx, prot_idx, protomer, screening_energy in screening_records:
            screen_delta = None
            if screening_energy is not None and min_screening_solution_energy is not None:
                screen_delta = screening_energy - min_screening_solution_energy
            if (
                screen_delta is not None
                and screen_delta > float(args.screen_threshold)
            ):
                screened_out.append((taut_idx, prot_idx, protomer, screening_energy, screen_delta))
            else:
                protomers_to_optimize.append((taut_idx, prot_idx, protomer, screening_energy, screen_delta))

        _log(
            "Screening finished: "
            f"kept={len(protomers_to_optimize)} "
            f"excluded={len(screened_out)} "
            f"threshold={float(args.screen_threshold):.2f} kcal/mol"
        )

        _log(" *** SOLVATION STAGE (g-xTB on screened geometry) ***")
        for taut_idx, prot_idx, protomer, _screening_energy, _screen_delta in protomers_to_optimize:
            protomer_items = list(spec.tautomers[taut_idx].protomers.items())
            prefix = (
                f"tautomer {taut_idx + 1}/{len(tautomer_items)} "
                f"protomer {prot_idx + 1}/{len(protomer_items)}"
            )
            _log(f"Optimizing {prefix}")
            _log(f"  [{prefix}] reusing screening conformer geometry (conformer_mode=skip_search)")
            run_protomer_solvation(
                protomer,
                protomer_id=str(prot_idx),
                scratch_root=species_scratch / f"tautomer_{taut_idx}",
                conformer_mode="skip_search",
                external_xyz_path=args.external_xyz,
                keep_scratch=bool(args.keep_scratch),
                keep_logs=bool(args.keep_logs),
                parse_solvation=args.parse_solvation,
                sp_energy=args.sp_energy,
                run_geometry_optimization=False,
                recompute_solvation=False,
                recompute_frequencies=False,
                reuse_screening_terms=True,
                dry_run=bool(args.dry_run),
                opt_level=args.opt_level,
                progress_callback=lambda stage, prefix=prefix: _log(f"  [{prefix}] {stage}"),
            )

        optimized_energies = [
            protomer.mol.GetDoubleProp("solution_phase_free_energy_kcal_mol")
            for _taut_idx, _prot_idx, protomer, _screening_energy, _screen_delta in protomers_to_optimize
            if protomer.mol is not None and protomer.mol.HasProp("solution_phase_free_energy_kcal_mol")
        ]
        min_postopt_solution_energy = min(optimized_energies) if optimized_energies else None
        if min_postopt_solution_energy is None:
            _log("No valid post-optimization solution energies found to backfill screened-out protomers.")

        for taut_idx, prot_idx, protomer, screening_energy, screen_delta in screened_out:
            if protomer.mol is None:
                continue
            protomer.mol.SetProp("screening_skipped_postopt", "true")
            _set_optional_double_prop(protomer, "screening_solution_phase_free_energy_kcal_mol", screening_energy)
            _set_optional_double_prop(protomer, "screening_delta_kcal_mol", screen_delta)
            placeholder_energy = None
            if min_postopt_solution_energy is not None and screen_delta is not None:
                placeholder_energy = min_postopt_solution_energy + screen_delta
            _set_optional_double_prop(protomer, "screening_placeholder_solution_phase_free_energy_kcal_mol", placeholder_energy)
            _set_optional_double_prop(protomer, "solution_phase_free_energy_kcal_mol", placeholder_energy)
            protomer.mol.SetProp("workflow_status", "screened_out")
            _log(
                "Screened out protomer "
                f"(tautomer {taut_idx + 1}, protomer {prot_idx + 1}) "
                f"screen_delta={screen_delta} "
                f"placeholder_solution_energy={placeholder_energy}"
            )
        _log(f"Optimization outputs saved under: {species_scratch}")

        _log("Calculating Boltzmann populations")
        excluded_unconverged_count = 0
        if args.exclude_unconverged:
            for taut in tautomer_items:
                for protomer in taut[1].protomers.values():
                    if (
                        protomer.mol is not None
                        and protomer.mol.HasProp("connectivity_mismatch")
                        and protomer.mol.GetProp("connectivity_mismatch").lower() == "true"
                    ):
                        excluded_unconverged_count += 1
            _log(
                "Excluding connectivity-mismatch protomers from Boltzmann weighting: "
                f"count={excluded_unconverged_count}"
            )
        spec.assign_boltzmann_microstate_populations(
            temperature_k=298.15,
            exclude_connectivity_mismatch=bool(args.exclude_unconverged),
        )
        f_zwit = spec.get_f_zwit()
        _log(f"Total predicted zwitterion fraction (f_zwit): {f_zwit:.5f}")
        for taut_idx, taut in tautomer_items:
            _log(f"  Tautomer {taut_idx + 1}/{len(tautomer_items)} Boltzmann populations:")
            for prot_idx, protomer in taut.protomers.items():
                frac = (
                    protomer.mol.GetProp("boltzmann_fraction")
                    if protomer.mol is not None and protomer.mol.HasProp("boltzmann_fraction")
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
