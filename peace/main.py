from peace.protomer import Species, Tautomer
from peace.engine import ChargeEngine
from peace.common import show_images, protonate_at_site, deprotonate_at_site
from peace import __version__
from datetime import datetime
import time
from pathlib import Path
from typing import Any, Optional
from rdkit.Chem import AllChem
import copy
import pandas as pd

def _build_cli_parser():
    import argparse

    p = argparse.ArgumentParser(description="PEACE demo: tautomer/protomer enumeration + optional xTB solvation workflow.")
    p.add_argument("--smiles", type=str, default="NCCCC(=O)CCC(=O)O", help="Input SMILES to enumerate.")
    p.add_argument(
        "--charge-min",
        type=int,
        default=0,
        help="Minimum formal charge state to include (inclusive).",
    )
    p.add_argument(
        "--charge-max",
        type=int,
        default=0,
        help="Maximum formal charge state to include (inclusive).",
    )
    p.add_argument(
        "--solvation",
        action="store_true",
        help="If set, runs a solvation energy workflow for all generated protomers (requires xTB + g-xTB binaries).",
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
    p.add_argument(
        "--refine",
        action="store_true",
        help=(
            "After the g-xTB solvation stage, re-evaluate solvation with ORCA openCOSMO-RS for "
            "low-lying protomers (requires ORCA on PATH). Implies workflows that completed with valid "
            "solution-phase energies; use with --solvation."
        ),
    )

    p.add_argument(
        "--refine-threshold",
        type=float,
        default=5.0,
        help=(
            "For --refine: include protomers whose g-xTB solution-phase free energy is within "
            "this many kcal/mol of the lowest such energy (default: 5)."
        ),
    )
    p.add_argument(
        "--orca-executable",
        type=str,
        default="orca",
        help="ORCA executable for the optional --refine COSMO-RS step.",
    )
    return p


def _make_species(smiles: str, *, engine: ChargeEngine) -> Species:
    spec = Species.from_smiles(smiles)
    tautomers = engine.search_for_tautomers(spec)
    spec.embed_tautomers_from_list_of_smiles(tautomers)

    return spec


def _enumerate_species_protomers(spec: Species, *, engine: ChargeEngine) -> None:
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


def _seed_adjacent_charge_species(
    source_spec: Species,
    *,
    engine: ChargeEngine,
    charge_step: int,
) -> Optional[Species]:
    if charge_step not in (-1, 1):
        raise ValueError("charge_step must be -1 or +1")

    search_type = "acidic" if charge_step < 0 else "basic"
    for _taut_idx, taut in source_spec.tautomers.items():
        for _prot_idx, protomer in taut.protomers.items():
            if protomer.mol is None:
                continue
            trial_taut = Tautomer.from_mol(copy.deepcopy(protomer.mol))
            sites = engine.search_ionization_centers(trial_taut, search_type)
            if not sites:
                continue
            selected_site = sites[0]
            shifted_mol = copy.deepcopy(protomer.mol)
            if charge_step < 0:
                atom = shifted_mol.GetAtomWithIdx(selected_site)
                local_h = atom.GetTotalNumHs(includeNeighbors=False)
                if local_h <= 0:
                    continue
                deprotonate_at_site(shifted_mol, selected_site)
            else:
                protonate_at_site(shifted_mol, selected_site)
            shifted_smiles = AllChem.MolToSmiles(shifted_mol)
            return _make_species(shifted_smiles, engine=engine)
    return None


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
    parser = _build_cli_parser()
    args = parser.parse_args()
    if int(args.charge_min) > int(args.charge_max):
        parser.error("--charge-min must be <= --charge-max")
    if args.refine and not args.solvation:
        parser.error("--refine requires --solvation")
    run_started_at = _ts()
    _log(_header_banner())
    _log(f"Version: {__version__}")
    _log(f"Run started at: {run_started_at}")
    _log(f"Input SMILES: {args.smiles}")
    _log(f"Requested formal charge range: [{int(args.charge_min)}, {int(args.charge_max)}]")

    engine = ChargeEngine()
    seed_spec = _make_species(args.smiles, engine=engine)
    seed_charge = AllChem.GetFormalCharge(seed_spec.tautomers[0].protomers[0].mol)
    _log(f"Input SMILES seed formal charge: {seed_charge}")
    _log(f"Generating seed Species at charge {seed_charge}")
    _enumerate_species_protomers(seed_spec, engine=engine)

    ############################
    # Charge seeding
    species_by_charge: dict[int, Species] = {int(seed_charge): seed_spec}
    charge_min = int(args.charge_min)
    charge_max = int(args.charge_max)

    # Search downwards from the provided structure charge.
    current_charge = int(seed_charge)
    current_spec = seed_spec
    while current_charge - 1 >= charge_min:
        target_charge = current_charge - 1
        _log(f"Attempting to seed charge state {target_charge} from {current_charge}")
        next_spec = _seed_adjacent_charge_species(current_spec, engine=engine, charge_step=-1)
        if next_spec is None:
            _log(
                f"No matching deprotonation found while searching charge {target_charge}; "
                "stopping lower-charge branch."
            )
            break
        _enumerate_species_protomers(next_spec, engine=engine)
        species_by_charge[target_charge] = next_spec
        current_spec = next_spec
        current_charge = target_charge

    # Search upwards from the provided structure charge.
    current_charge = int(seed_charge)
    current_spec = seed_spec
    while current_charge + 1 <= charge_max:
        target_charge = current_charge + 1
        _log(f"Attempting to seed charge state {target_charge} from {current_charge}")
        next_spec = _seed_adjacent_charge_species(current_spec, engine=engine, charge_step=1)
        if next_spec is None:
            _log(
                f"No matching protonation found while searching charge {target_charge}; "
                "stopping higher-charge branch."
            )
            break
        _enumerate_species_protomers(next_spec, engine=engine)
        species_by_charge[target_charge] = next_spec
        current_spec = next_spec
        current_charge = target_charge

    requested_charges = [c for c in sorted(species_by_charge.keys()) if charge_min <= c <= charge_max]
    if not requested_charges:
        _log("No species generated inside the requested charge window.")

    ######################
    # Solvation workflow

    if args.solvation:
        from peace.solvation import run_protomer_solvation, run_protomer_screening
        import shutil

        scratch_root_path = Path(args.scratch_root)
        for charge_state in requested_charges:
            spec = species_by_charge[charge_state]
            tautomer_items = list(spec.tautomers.items())
            species_scratch = scratch_root_path / f"charge_{charge_state:+d}" / f"species_{spec.key}"
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

            _log(f" *** SCREENING PROTOMERS (charge={charge_state}) *** ")
            screening_records: list[tuple[int, int, Any, Optional[float]]] = []
            for taut_idx, taut in tautomer_items:
                protomer_items = list(taut.protomers.items())
                for prot_idx, protomer in protomer_items:
                    prefix = (
                        f"charge {charge_state:+d} "
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
                    f"charge {charge_state:+d} "
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
                    sp_energy=args.sp_energy,
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
            if args.refine:
                from peace.solvation import refine_protomer_solvation_with_orca_cosmors

                _log(" *** ORCA openCOSMO-RS REFINE *** ")
                refine_pool: list[tuple[int, int, Any, float]] = []
                for taut_idx, prot_idx, protomer, _screen_e, _screen_d in protomers_to_optimize:
                    if protomer.mol is None or not protomer.mol.HasProp("solution_phase_free_energy_kcal_mol"):
                        continue
                    try:
                        g_sol = float(protomer.mol.GetProp("solution_phase_free_energy_kcal_mol"))
                    except ValueError:
                        continue
                    refine_pool.append((taut_idx, prot_idx, protomer, g_sol))
                if not refine_pool:
                    _log("Refine: no protomers with valid post–g-xTB solution-phase energies; skipping ORCA.")
                else:
                    g_min_ref = min(x[3] for x in refine_pool)
                    thr_ref = float(args.refine_threshold)
                    picked = [x for x in refine_pool if x[3] - g_min_ref <= thr_ref + 1e-9]
                    if len(picked) < 2:
                        _log(
                            "Refine: fewer than two protomers within threshold; skipping openCOSMO-RS ORCA calc to save resources. "
                            f"within_threshold={len(picked)} threshold={thr_ref:.3f} kcal/mol"
                        )
                        _log(
                            "Refine: criterion requires at least two candidates (e.g., 0 and <= threshold kcal/mol) "
                            "relative to the lowest post–g-xTB solution-phase energy."
                        )
                        picked = []
                    picked_keys = {(taut_idx, prot_idx) for taut_idx, prot_idx, _protomer, _g_sol in picked}
                    _log(
                        "Refine: lowest g-xTB solution-phase G = "
                        f"{g_min_ref:.6f} kcal/mol; threshold = {thr_ref:.3f} kcal/mol; "
                        f"ORCA COSMO-RS runs = {len(picked)}"
                    )
                    for taut_idx, prot_idx, protomer, g_sol in picked:
                        n_prot = len(spec.tautomers[taut_idx].protomers)
                        prefix = (
                            f"charge {charge_state:+d} "
                            f"tautomer {taut_idx + 1}/{len(tautomer_items)} "
                            f"protomer {prot_idx + 1}/{n_prot}"
                        )
                        delta = g_sol - g_min_ref
                        _log(f"ORCA refine {prefix} (ΔG from min = {delta:.3f} kcal/mol)")
                        refine_root = species_scratch / f"tautomer_{taut_idx}" / "orca_refine" / f"protomer_{prot_idx}"
                        wf_log = species_scratch / "peace.out"
                        try:
                            refine_protomer_solvation_with_orca_cosmors(
                                protomer,
                                scratch_dir=refine_root,
                                solvent="water",
                                keep_scratch=bool(args.keep_scratch),
                                orca_executable=str(args.orca_executable),
                                dry_run=bool(args.dry_run),
                                log_paths=[wf_log],
                                progress_callback=lambda s, p=prefix: _log(f"  [{p}] {s}"),
                            )
                        except Exception as exc:
                            _log(f"ORCA refine failed for {prefix}: {exc}")

                    refined_success: list[tuple[int, int, Any, float]] = []
                    for taut_idx, prot_idx, protomer, _g_sol_pre in picked:
                        if protomer.mol is None:
                            continue
                        if (
                            protomer.mol.HasProp("solvation_refined_cosmors")
                            and protomer.mol.GetProp("solvation_refined_cosmors").lower() == "true"
                            and protomer.mol.HasProp("solution_phase_free_energy_kcal_mol")
                        ):
                            try:
                                g_refined = float(protomer.mol.GetProp("solution_phase_free_energy_kcal_mol"))
                            except ValueError:
                                continue
                            refined_success.append((taut_idx, prot_idx, protomer, g_refined))

                    if not refined_success:
                        _log("Refine: no successful ORCA results; leaving post–g-xTB energies unchanged.")
                    else:
                        min_refined_final = min(x[3] for x in refined_success)
                        _log(
                            "Refine propagation: anchoring non-refined screened-in protomers to "
                            f"lowest refined final G = {min_refined_final:.6f} kcal/mol using pre-refine deltas."
                        )

                        for taut_idx, prot_idx, protomer, g_sol_pre in refine_pool:
                            if protomer.mol is None:
                                continue
                            is_refined_success = (
                                protomer.mol.HasProp("solvation_refined_cosmors")
                                and protomer.mol.GetProp("solvation_refined_cosmors").lower() == "true"
                            )
                            if is_refined_success:
                                continue
                            delta_pre = g_sol_pre - g_min_ref
                            propagated_energy = min_refined_final + delta_pre
                            protomer.mol.SetDoubleProp("refine_pre_delta_kcal_mol", float(delta_pre))
                            protomer.mol.SetDoubleProp(
                                "solution_phase_free_energy_unrefined_kcal_mol",
                                float(g_sol_pre),
                            )
                            protomer.mol.SetDoubleProp(
                                "solution_phase_free_energy_kcal_mol",
                                float(propagated_energy),
                            )
                            protomer.mol.SetProp("refine_propagated_from_unrefined_delta", "true")
                            status = "selected_but_refine_failed" if (taut_idx, prot_idx) in picked_keys else "not_selected_for_refine"
                            protomer.mol.SetProp("refine_status", status)

                        # Keep screened-out placeholder energies consistent with the final refined baseline.
                        for taut_idx, prot_idx, protomer, _screening_energy, screen_delta in screened_out:
                            if protomer.mol is None or screen_delta is None:
                                continue
                            placeholder_energy = min_refined_final + screen_delta
                            _set_optional_double_prop(
                                protomer,
                                "screening_placeholder_solution_phase_free_energy_kcal_mol",
                                placeholder_energy,
                            )
                            _set_optional_double_prop(protomer, "solution_phase_free_energy_kcal_mol", placeholder_energy)
                            protomer.mol.SetProp("screening_placeholder_reanchored_to_refined_min", "true")

            _log(f"Optimization outputs saved under: {species_scratch}")

            _log(f"Calculating Boltzmann populations for charge={charge_state}")
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
            _log(f"Total predicted zwitterion fraction (f_zwit) for charge={charge_state}: {f_zwit:.5f}")
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
        for charge_state in requested_charges:
            spec = species_by_charge[charge_state]
            _log(f"Rendering protomer plots for charge={charge_state}")
            imgs = spec.generate_protomer_plot(n_columns=5)
            show_images(imgs, mode="vertical")
    frames = []
    for charge_state in requested_charges:
        spec = species_by_charge[charge_state]
        frame = spec.to_dataframe()
        frame["formal_charge"] = int(charge_state)
        frames.append(frame)
    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame()
    print(df)

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        _log(f"Saved dataframe CSV to: {output_path.resolve()}")

    end_ts = time.time()
    _log(f"Run finished at: {_ts()}")
    _log(f"Execution time: {end_ts - start_ts:.2f} s")
