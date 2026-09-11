import common


def main() -> None:
    args = common.parse_args("Appendix A4: r5 at 2.0 m/s on the finer and coarser meshes")
    root = common.eval_root(args.output_root)
    sims = common.select(common.MESH_SIMS, args.sims)
    common.evaluate_models(
        common.select(common.MAIN_MODELS, args.models),
        common.select(common.SEEDS, args.seeds),
        sims,
        root,
        starts=common.STARTS[:1],
        keep_fields=False,
    )
    common.evaluate_models(
        common.select(common.CONTROL_MODELS, args.models),
        common.SEEDS[:1],
        sims,
        root,
        starts=common.STARTS[:1],
        keep_fields=False,
    )


if __name__ == "__main__":
    main()
