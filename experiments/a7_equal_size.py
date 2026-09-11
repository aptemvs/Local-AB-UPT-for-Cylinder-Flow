import common


def main() -> None:
    args = common.parse_args("Appendix A7: global models of the same size on Experiments 1 and 2")
    common.evaluate_models(
        common.select(common.CONTROL_MODELS, args.models),
        common.SEEDS[:1],
        common.select(common.SPEED_SIMS + common.INTERPOLATION_SIMS + common.POS_SIMS, args.sims),
        common.eval_root(args.output_root),
        starts=common.STARTS[:1],
        keep_fields=False,
    )


if __name__ == "__main__":
    main()
