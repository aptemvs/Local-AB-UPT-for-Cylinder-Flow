import common

DEPTHS = range(1, 13)


def main() -> None:
    args = common.parse_args("Appendix A2: global stacks s^1 to s^12 on the validation flow")
    root = common.eval_root(args.output_root)
    models = [common.Model(f"g{n}", "s" * n) for n in DEPTHS]
    sims = common.select((common.VALIDATION_SIM, common.TEST_SIM), args.sims)
    for model in common.select(models, args.models):
        run_dir = common.ensure_model_run(model, common.SEEDS[0])
        for sim in sims:
            common.evaluate_depth(model.name, run_dir, sim, root)


if __name__ == "__main__":
    main()
