import common


def main() -> None:
    args = common.parse_args("Train s8, l5 and lllsss on exp40 for three seeds")
    for model in common.select(common.MAIN_MODELS, args.models):
        for seed in common.select(common.SEEDS, args.seeds):
            run_dir = common.ensure_model_run(model, seed)
            print(f"{model.name} seed {seed}: {run_dir.relative_to(common.ROOT)}")
    common.write_truth_stats(
        common.SPEED_SIMS + common.POS_SIMS + common.TRAIN_SIMS + common.INTERPOLATION_SIMS,
        common.eval_root(args.output_root),
    )


if __name__ == "__main__":
    main()
