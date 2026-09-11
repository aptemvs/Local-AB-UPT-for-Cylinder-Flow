import common

RADIUS = 0.1
ORDERS = (
    common.Model("g6", "ssssss"),
    common.Model("l6", "llllll", RADIUS),
    common.Model("ssslll", "ssslll", RADIUS),
    common.Model("slslsl", "slslsl", RADIUS),
    common.Model("lllsss", "lllsss", RADIUS),
)
LADDER = tuple(
    common.Model(blocks, blocks, RADIUS)
    for blocks in ("llllsss", "llllssss", "lllllssss", "lllllsssss", "llllllsssss")
)


def main() -> None:
    args = common.parse_args("Appendix A3: block order at six blocks and deeper local-first stacks")
    root = common.eval_root(args.output_root)
    sims = common.select((common.VALIDATION_SIM, common.TEST_SIM), args.sims)
    for model in common.select(ORDERS + LADDER, args.models):
        run_dir = common.ensure_model_run(model, common.SEEDS[0])
        for sim in sims:
            common.evaluate_depth(model.name, run_dir, sim, root)


if __name__ == "__main__":
    main()
