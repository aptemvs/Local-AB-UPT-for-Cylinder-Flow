import json
import statistics
import time

import common
import torch

from abupt.rollout import rollout

COST_DIR = "thesis_cost"
SIM = "r5_u2.3"
START = 201
REPEATS = 5
WARMUP_FRAMES = 22
COST_STEPS = 300
COST_EVAL_INTERVAL = 100


def timed_rollout(
    model: torch.nn.Module, warm: dict[str, torch.Tensor], num_anchors: int, seed: int
) -> float:
    torch.cuda.synchronize(common.DEVICE)
    started = time.perf_counter()
    rollout(model, warm, num_anchors, seed)
    torch.cuda.synchronize(common.DEVICE)
    return time.perf_counter() - started


def main() -> None:
    args = common.parse_args("Computational cost of the main models and the equal-size controls")
    cost_dir = (args.output_root or common.OUTPUTS) / COST_DIR
    cost_dir.mkdir(parents=True, exist_ok=True)
    path = cost_dir / "cost.json"
    cost = json.loads(path.read_text()) if path.exists() else {}
    cost.update(gpu=torch.cuda.get_device_name(0), sim=SIM, start=START, repeats=REPEATS)
    for model in common.select(common.MAIN_MODELS + common.CONTROL_MODELS, args.models):
        if model.name in cost:
            continue
        seeds = common.SEEDS if model in common.MAIN_MODELS else common.SEEDS[:1]
        runs = [common.ensure_model_run(model, seed) for seed in seeds]
        metrics = [json.loads((run / "metrics.json").read_text()) for run in runs]
        cost_run = common.ensure_run(
            f"cost-{model.name}",
            [
                *common.model_overrides(model),
                f"trainer.max_steps={COST_STEPS}",
                f"trainer.val_check_interval={COST_EVAL_INTERVAL}",
            ],
        )
        peak_train = json.loads((cost_run / "metrics.json").read_text())["peak_mem_bytes"]

        network, hparams = common.load_model(common.checkpoint_path(runs[0]))
        num_anchors, seed = hparams["num_anchors"], hparams["seed"]
        traj = common.load_sim(SIM)
        warm = {**traj, "velocity": traj["velocity"][START - 1 :]}
        steps = warm["velocity"].shape[0] - 2
        timed_rollout(
            network, {**warm, "velocity": warm["velocity"][:WARMUP_FRAMES]}, num_anchors, seed
        )
        torch.cuda.reset_peak_memory_stats()
        seconds = [timed_rollout(network, warm, num_anchors, seed) for _ in range(REPEATS)]
        train_min = [m["train_seconds"] / 60 for m in metrics]
        cost[model.name] = {
            "blocks": hparams["blocks"],
            "depth": len(hparams["blocks"]),
            "num_params": metrics[0]["num_params"],
            "seeds": [str(run.relative_to(common.ROOT)) for run in runs],
            "train_min": train_min,
            "train_min_mean": statistics.mean(train_min),
            "train_min_std": statistics.stdev(train_min) if len(train_min) > 1 else 0.0,
            "peak_train_mem_bytes": peak_train,
            "peak_train_run": str(cost_run.relative_to(common.ROOT)),
            "rollout_steps": int(steps),
            "rollout_seconds": seconds,
            "rollout_seconds_median": statistics.median(seconds),
            "rollout_ms_per_step": 1000 * statistics.median(seconds) / steps,
            "peak_rollout_mem_bytes": torch.cuda.max_memory_allocated(),
        }
        path.write_text(json.dumps(cost, indent=1))
        print(
            f"{model.name:8s} params {cost[model.name]['num_params']:,} "
            f"train {cost[model.name]['train_min_mean']:.1f} min "
            f"rollout {cost[model.name]['rollout_ms_per_step']:.2f} ms/step",
            flush=True,
        )


if __name__ == "__main__":
    main()
