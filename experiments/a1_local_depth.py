import common

DEPTHS = range(1, 13)
RADIUS = 0.1


def main() -> None:
    args = common.parse_args("Appendix A1: local stacks l^1 to l^12 on the validation flow")
    root = common.eval_root(args.output_root)
    models = [common.Model(f"l{n}", "l" * n, RADIUS) for n in DEPTHS]
    sims = common.select((common.VALIDATION_SIM, common.TEST_SIM), args.sims)
    for model in common.select(models, args.models):
        run_dir = common.ensure_model_run(model, common.SEEDS[0])
        for sim in sims:
            common.evaluate_depth(model.name, run_dir, sim, root)


if __name__ == "__main__":
    main()
