import os
from glob import glob

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from abupt.dataset import Dataset, load_trajectory


def load_split(data_dir: str, split: str) -> list[dict[str, torch.Tensor]]:
    paths = sorted(glob(os.path.join(data_dir, split, "traj_*.npz")))
    if not paths:
        raise FileNotFoundError(f"no trajectories under {data_dir}/{split}")
    return [load_trajectory(path) for path in paths]


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        num_anchors: int,
        num_queries: int,
        noise_scale: float,
        target: str,
        num_val_windows: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_anchors = num_anchors
        self.num_queries = num_queries
        self.noise_scale = noise_scale
        self.target = target
        self.num_val_windows = num_val_windows
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        self.train_trajectories = load_split(self.data_dir, "train")
        self.valid_trajectories = load_split(self.data_dir, "valid")
        common = dict(
            num_anchors=self.num_anchors,
            num_queries=self.num_queries,
            noise_scale=self.noise_scale,
            target=self.target,
            seed=self.seed,
        )
        self.train = Dataset(self.train_trajectories, train=True, **common)
        self.valid = Dataset(self.valid_trajectories, train=False, **common)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        count = min(self.num_val_windows, len(self.valid))
        indices = np.unique(np.linspace(0, len(self.valid) - 1, count).astype(int)).tolist()
        return DataLoader(Subset(self.valid, indices), batch_size=1, num_workers=self.num_workers)
