from peace.protomer import Species
from peace.engine import ChargeEngine
from peace.common import show_images
from rdkit.Chem import Draw

def _build_cli_parser():
    import argparse

    p = argparse.ArgumentParser(description="PEACE demo: tautomer/protomer enumeration + optional xTB solvation workflow.")
    p.add_argument("--smiles", type=str, default="NCCCC(=O)CCC(=O)O", help="Input SMILES to enumerate.")
    p.add_argument(
        "--optimize",
        action="store_true",
        help="If set, run the solvation energy workflow for all generated protomers (requires xTB + g-xTB binaries).",
    )
    p.add_argument("--scratch-root", type=str, default="./peace_scratch_solvation", help="Scratch root for xTB runs.")
    p.add_argument("--keep-scratch", action="store_true", help="Keep xTB scratch directories after each protomer run.")
    p.add_argument("--no-plot", action="store_true", help="Skip RDKit image rendering.")
    p.add_argument(
        "--conformer-mode",
        type=str,
        default="mmff94",
        choices=["mmff94", "external_xyz", "skip_search"],
        help="Conformer geometry input for xTB runs.",
    )
    p.add_argument("--external-xyz", type=str, default=None, help="Path to external xyz (used only with conformer-mode=external_xyz).")
    p.add_argument("--dry-run", action="store_true", help="Do not execute xTB; still run embedding/IO.")
    return p


def _run_demo(smiles: str, *, no_plot: bool) -> Species:
    spec = Species.from_smiles(smiles)
    engine = ChargeEngine()

    tautomers = engine.search_for_tautomers(spec)
    print(tautomers)  # list of SMILES. Could be provided manually if desired.
    spec.embed_tautomers_from_list_of_smiles(tautomers)

    # Generate protomers for each tautomer based on available acid/base sites.
    for taut in spec.tautomers.values():
        acid_sites = engine.search_ionization_centers(taut, "strong_acidic")
        basic_sites = engine.search_ionization_centers(taut, "strong_basic")
        taut.generate_protomers_from_base_protomer(acid_sites, basic_sites)

    if not no_plot:
        imgs = spec.generate_protomer_plot(n_columns=5)
        show_images(imgs, mode="vertical")
    print(spec.to_dataframe())
    return spec


if __name__ == "__main__":
    args = _build_cli_parser().parse_args()
    spec = _run_demo(args.smiles, no_plot=bool(args.no_plot))

    if args.optimize:
        from peace.solvation import run_species_solvation

        run_species_solvation(
            spec,
            scratch_root=args.scratch_root,
            conformer_mode=args.conformer_mode,
            external_xyz_path=args.external_xyz,
            keep_scratch=bool(args.keep_scratch),
            hess_charge_mode=args.hess_charge_mode,
            dry_run=bool(args.dry_run),
        )