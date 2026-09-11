import json
from pathlib import Path

import common
import numpy as np

from abupt.rollout import rollout, rollout_metrics

RADII = (0.05, 0.1, 0.2, 0.4, 0.8)
PROBE_STEPS = 100
TRAIN_SIM = "r5_u2.0"
CASES = {"exp00": ("r5_u2.0", "r5_u2.0"), "exp15": ("r6_u2.0", "r8_u2.0")}


def evaluate_case(run_dir: Path, val_sim: str, test_sim: str) -> tuple[dict, dict]:
    paths = sorted((run_dir / "ckpts").glob("step_*.*"))
    val_traj, test_traj = common.load_sim(val_sim), common.load_sim(test_sim)

    selection = []
    for path in paths:
        model, hparams = common.load_model(path)
        errors, _ = rollout(model, val_traj, hparams["num_anchors"], hparams["seed"], PROBE_STEPS)
        rmse50 = float(np.sqrt(np.mean(errors[:50])))
        selection.append({"step": common.checkpoint_step(path), "probe_rmse50": rmse50})
    best_index = int(np.argmin([row["probe_rmse50"] for row in selection]))

    results = {}
    for stem, index in (("best", best_index), ("last", len(paths) - 1)):
        model, hparams = common.load_model(paths[index])
        errors, _ = rollout(model, test_traj, hparams["num_anchors"], hparams["seed"])
        metrics = rollout_metrics([errors])
        results[stem] = {key: metrics[key] for key in ("rmse_1", "rmse_50", "rmse_all")}
        results[stem]["step"] = common.checkpoint_step(paths[index])
        results[stem]["mse_per_step"] = metrics["mse_per_step"]

    summary = {
        "triple": f"{TRAIN_SIM} / {val_sim} / {test_sim}",
        "trainer_run": str(run_dir.relative_to(common.ROOT)),
        "best_step": selection[best_index]["step"],
        "best_probe_rmse50": selection[best_index]["probe_rmse50"],
        "best": {k: v for k, v in results["best"].items() if k != "mse_per_step"},
        "last": {k: v for k, v in results["last"].items() if k != "mse_per_step"},
    }
    detail = {
        **summary,
        "selection_curve": selection,
        "best_mse_per_step": results["best"]["mse_per_step"],
        "last_mse_per_step": results["last"]["mse_per_step"],
    }
    return summary, detail


def write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "| exp | train/val/test | best step | best rmse_1 | best rmse_50 | best rmse_all "
        "| last rmse_all |",
        "|---|---|---|---|---|---|---|",
    ]
    for exp in sorted(summary):
        row = summary[exp]
        best, last = row["best"], row["last"]
        lines.append(
            f"| {exp} | {row['triple']} | {row['best_step']} "
            f"| {best['rmse_1']:.5f} | {best['rmse_50']:.3f} | {best['rmse_all']:.3f} "
            f"| {last['rmse_all']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = common.parse_args("Appendix A5: locality radius of six local blocks, early recipe")
    root = args.output_root or common.OUTPUTS
    for radius in common.select(RADII, args.models):
        run_dir = common.ensure_run(
            f"radius-r{radius:g}", [f"model.local_radius={radius}"], config_name="radius"
        )
        battery = root / f"comsol_battery_llllll_r{radius:g}"
        battery.mkdir(parents=True, exist_ok=True)
        summary_path = battery / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        updated = False
        for exp in common.select(CASES, args.sims):
            if exp in summary:
                continue
            val_sim, test_sim = CASES[exp]
            summary[exp], detail = evaluate_case(run_dir, val_sim, test_sim)
            (battery / f"{exp}_detail.json").write_text(json.dumps(detail, indent=2))
            updated = True
            print(
                f"r={radius:g} {exp}: best@{summary[exp]['best_step']} "
                f"rmse_all {summary[exp]['best']['rmse_all']:.3e} "
                f"(last {summary[exp]['last']['rmse_all']:.3e})",
                flush=True,
            )
        if updated or not (battery / "summary.md").exists():
            summary_path.write_text(json.dumps(summary, indent=2))
            write_markdown(summary, battery / "summary.md")


if __name__ == "__main__":
    main()
