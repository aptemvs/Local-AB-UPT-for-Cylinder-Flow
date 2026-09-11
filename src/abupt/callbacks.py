import json
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch

from abupt.rollout import rollout


class ProbeRolloutCallback(L.Callback):
    def __init__(self, num_trajectories: int = 2, num_steps: int = 100) -> None:
        self.num_trajectories = num_trajectories
        self.num_steps = num_steps

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        hparams = pl_module.hparams
        trajectories = trainer.datamodule.valid_trajectories[: self.num_trajectories]
        curves = np.asarray(
            [
                rollout(pl_module.model, traj, hparams.num_anchors, hparams.seed, self.num_steps)[0]
                for traj in trajectories
            ]
        )
        pl_module.log(
            "validation/probe_rmse50", float(np.sqrt(curves[:, :50].mean())), prog_bar=True
        )
        pl_module.log("validation/probe_rmse", float(np.sqrt(curves.mean())))


class RunMetricsCallback(L.Callback):
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.started = 0.0

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.started = time.time()
        if trainer.strategy.root_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        device = trainer.strategy.root_device
        checkpoint = trainer.checkpoint_callback
        best_step = None
        if checkpoint.best_model_path:
            best_step = torch.load(
                checkpoint.best_model_path, map_location="cpu", weights_only=True
            )["global_step"]
        metrics = {
            "num_params": sum(p.numel() for p in pl_module.parameters()),
            "train_seconds": time.time() - self.started,
            "peak_mem_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "best_probe_rmse50": (
                checkpoint.best_model_score.item()
                if checkpoint.best_model_score is not None
                else None
            ),
            "best_step": best_step,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(metrics, indent=2))
