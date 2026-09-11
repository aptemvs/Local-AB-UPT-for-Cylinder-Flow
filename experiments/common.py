import argparse
import functools
import gc
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from abupt.dataset import load_trajectory
from abupt.evaluation import (
    MeanReference,
    one_step_rmse,
    rollout_scalars,
    train_means_on,
    training_reference,
    truth_stats,
)
from abupt.lit_model import load_network
from abupt.nn import AnchoredBranchedUPT

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
POOL = ROOT / "data" / "comsol_cylinder" / "pool"
EVAL_DIR = "thesis_eval"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAST_STEP = 200000
STARTS = (201, 301, 401)
SEEDS = (42, 43, 44)
SPEEDS = (
    "2.0",
    "2.1",
    "2.2",
    "2.3",
    "2.4",
    "2.5",
    "2.6",
    "2.7",
    "2.8",
    "2.9",
    "3.0",
    "3.1",
    "3.2",
    "3.3",
    "3.6",
    "4.0",
)
SPEED_SIMS = [f"r5_u{u}" for u in SPEEDS]
POS_SIMS = [f"r{i}_u2.2" for i in range(1, 16)]
TRAIN_SIMS = [f"{r}_u{u}" for r in ("r4", "r5", "r6") for u in ("2.0", "2.1", "2.2", "2.5")]
INTERPOLATION_SIMS = ["r4_u2.3", "r6_u2.4"]
MESH_SIMS = ["r5_u2.0_meshB", "r5_u2.0_meshC"]
VALIDATION_SIM = "r5_u2.4"
TEST_SIM = "r5_u2.3"


@dataclass(frozen=True)
class Model:
    name: str
    blocks: str
    local_radius: float | None = None


MAIN_MODELS = (
    Model("s8", "ssssssss"),
    Model("l5", "lllll", 0.1),
    Model("lllsss", "lllsss", 0.1),
)
CONTROL_MODELS = (Model("s5", "sssss"), Model("s6", "ssssss"))


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--sims", nargs="*", default=None)
    return parser.parse_args()


def select(items: Iterable[Any], chosen: Sequence[Any] | None) -> list[Any]:
    if chosen is None:
        return list(items)
    keys = {str(item) for item in chosen}
    return [item for item in items if str(getattr(item, "name", item)) in keys]


def eval_root(output_root: Path | None) -> Path:
    root = (output_root or OUTPUTS) / EVAL_DIR
    for name in ("scalars", "fields", "depth"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def run_label(model: Model, seed: int) -> str:
    radius = "" if model.local_radius is None else f"-r{model.local_radius:g}"
    return f"{model.blocks}{radius}-seed{seed}"


def model_overrides(model: Model) -> list[str]:
    overrides = [f"model.blocks={model.blocks}"]
    if model.local_radius is not None:
        overrides.append(f"model.local_radius={model.local_radius}")
    return overrides


def train(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision(cfg.extras.float32_matmul_precision)
    L.seed_everything(cfg.extras.seed, workers=True)
    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    logger = instantiate(cfg.logger)
    callbacks = [instantiate(callback) for callback in cfg.callbacks.values()]
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule)
    del trainer, model, datamodule
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()


def ensure_run(label: str, overrides: Sequence[str], config_name: str = "train") -> Path:
    run_dir = OUTPUTS / label
    if not any(run_dir.glob("last.*")):
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
            cfg = compose(config_name, overrides=[*overrides, f"paths.output_dir={run_dir}"])
        print(f"training {label} -> {run_dir}", flush=True)
        train(cfg)
    return run_dir


def ensure_model_run(model: Model, seed: int) -> Path:
    return ensure_run(run_label(model, seed), [*model_overrides(model), f"extras.seed={seed}"])


def checkpoint_path(run_dir: Path, step: int = LAST_STEP) -> Path:
    return next(run_dir.glob(f"ckpts/step_{step:07d}.*"))


def checkpoint_step(path: Path) -> int:
    return int(path.stem.split("_")[1])


def load_model(path: Path) -> tuple[AnchoredBranchedUPT, dict[str, Any]]:
    return load_network(path, DEVICE)


@functools.cache
def load_sim(sim: str) -> dict[str, torch.Tensor]:
    return load_trajectory(str(POOL / f"{sim}.npz"))


@functools.cache
def reference() -> tuple[MeanReference, float]:
    return training_reference({sim: load_sim(sim) for sim in TRAIN_SIMS})


@functools.cache
def train_means(sim: str) -> dict[str, np.ndarray]:
    return train_means_on(load_sim(sim)["mesh_pos"].numpy(), reference()[0])


def write_truth_stats(sims: Iterable[str], root: Path) -> None:
    path = root / "truth_stats.json"
    stats = json.loads(path.read_text()) if path.exists() else {}
    missing = [sim for sim in sims if sim not in stats]
    if not missing and "_train_max_speed" in stats:
        return
    train_max_speed = reference()[1]
    for sim in missing:
        stats[sim] = truth_stats(sim, load_sim(sim), train_max_speed)
    stats["_train_max_speed"] = train_max_speed
    path.write_text(json.dumps(stats, indent=1))


def evaluate(name: str, run_dir: Path, sim: str, start: int, keep_fields: bool, root: Path) -> None:
    out_path = root / "scalars" / f"{name}__{sim}__s{start}.json"
    if out_path.exists():
        return
    model, hparams = load_model(checkpoint_path(run_dir))
    traj = load_sim(sim)
    num_anchors, seed = hparams["num_anchors"], hparams["seed"]
    scalars, fields = rollout_scalars(
        model, traj, num_anchors, seed, start, train_means(sim), reference()[1]
    )
    if start == STARTS[0]:
        scalars["one_step_rmse"] = one_step_rmse(model, traj, num_anchors, seed)
        if keep_fields:
            np.savez_compressed(
                root / "fields" / f"{name}__{sim}__s{start}.npz", fields=fields.astype(np.float16)
            )
    scalars.update(model=name, run_dir=str(run_dir.relative_to(ROOT)), step=LAST_STEP, sim=sim)
    out_path.write_text(json.dumps(scalars, indent=1))
    print(
        f"{name:12s} {sim:14s} s{start}: rmse_50 {scalars['rmse_50']:.4f} "
        f"rmse_all {scalars['rmse_all']:.4f} "
        f"1step {scalars.get('one_step_rmse', float('nan')):.5f} "
        f"f {scalars['peak_freq_pred']:.2f}/{scalars['peak_freq_truth']:.2f}",
        flush=True,
    )


def evaluate_depth(tag: str, run_dir: Path, sim: str, root: Path) -> None:
    out_path = root / "depth" / f"{tag}__{sim}.json"
    if out_path.exists():
        return
    model, hparams = load_model(checkpoint_path(run_dir))
    traj = load_sim(sim)
    num_anchors, seed = hparams["num_anchors"], hparams["seed"]
    scalars, _ = rollout_scalars(
        model, traj, num_anchors, seed, STARTS[0], train_means(sim), reference()[1]
    )
    scalars["one_step_rmse"] = one_step_rmse(model, traj, num_anchors, seed)
    scalars.update(
        model=tag,
        run_dir=str(run_dir.relative_to(ROOT)),
        step=LAST_STEP,
        sim=sim,
        blocks=hparams["blocks"],
        params=sum(p.numel() for p in model.parameters()),
    )
    out_path.write_text(json.dumps(scalars, indent=1))
    print(
        f"depth {tag:12s} {sim}: rmse_50 {scalars['rmse_50']:.4f} "
        f"rmse_all {scalars['rmse_all']:.4f} 1step {scalars['one_step_rmse']:.5f}",
        flush=True,
    )


def evaluate_models(
    models: Iterable[Model],
    seeds: Iterable[int],
    sims: Sequence[str],
    root: Path,
    starts: Sequence[int] = STARTS,
    keep_fields: bool = True,
) -> None:
    for model in models:
        for seed in seeds:
            run_dir = ensure_model_run(model, seed)
            name = model.name if seed == SEEDS[0] else f"{model.name}_seed{seed}"
            for sim in sims:
                evaluate(name, run_dir, sim, starts[0], keep_fields and seed == SEEDS[0], root)
            if seed == SEEDS[0]:
                for start in starts[1:]:
                    for sim in sims:
                        evaluate(name, run_dir, sim, start, False, root)
