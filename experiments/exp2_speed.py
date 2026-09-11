import common


def main() -> None:
    args = common.parse_args(
        "Experiment 2: inlet speeds at r5 and interpolation tests at r4 and r6"
    )
    common.evaluate_models(
        common.select(common.MAIN_MODELS, args.models),
        common.select(common.SEEDS, args.seeds),
        common.select(common.SPEED_SIMS + common.INTERPOLATION_SIMS, args.sims),
        common.eval_root(args.output_root),
    )


if __name__ == "__main__":
    main()
