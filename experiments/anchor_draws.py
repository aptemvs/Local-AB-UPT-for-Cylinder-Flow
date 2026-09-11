import json

import common
import numpy as np

from abupt.rollout import rollout

CASES = ("r5_u2.3", "r5_u2.7", "r15_u2.2", "r1_u2.2")
ANCHOR_SEEDS = range(1000, 1010)


def main() -> None:
    args = common.parse_args("Variation over anchor draws for the seed-42 main models")
    out_path = common.eval_root(args.output_root) / "anchors_all.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for model in common.select(common.MAIN_MODELS, args.models):
        run_dir = common.ensure_model_run(model, common.SEEDS[0])
        network, hparams = common.load_model(common.checkpoint_path(run_dir))
        for sim in common.select(CASES, args.sims):
            key = f"{model.name}__{sim}"
            if key in out:
                continue
            traj = common.load_sim(sim)
            warm = {**traj, "velocity": traj["velocity"][common.STARTS[0] - 1 :]}
            record = {"rmse_all": [], "rmse_50": [], "steps": []}
            for anchor_seed in ANCHOR_SEEDS:
                errors = np.asarray(rollout(network, warm, hparams["num_anchors"], anchor_seed)[0])
                record["rmse_all"].append(float(np.sqrt(errors.mean())))
                record["rmse_50"].append(float(np.sqrt(errors[:50].mean())))
                record["steps"].append(int(len(errors)))
            out[key] = record
            out_path.write_text(json.dumps(out, indent=1))
            values = np.asarray(record["rmse_all"])
            print(
                f"{model.name:8s} {sim:9s} rmse_all {values.mean():.4f} "
                f"+- {values.std(ddof=1):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
