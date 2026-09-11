import common


def main() -> None:
    args = common.parse_args("Experiment 1: cylinder positions r1 to r15 at 2.2 m/s")
    common.evaluate_models(
        common.select(common.MAIN_MODELS, args.models),
        common.select(common.SEEDS, args.seeds),
        common.select(common.POS_SIMS, args.sims),
        common.eval_root(args.output_root),
    )


if __name__ == "__main__":
    main()
