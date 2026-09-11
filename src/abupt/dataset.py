import numpy as np
import torch
import torch.nn.functional as F

NUM_NODE_TYPES = 9
NORMAL_NODE = 0
PREDICTED_NODES = (0, 5)


def load_trajectory(path: str) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        traj = {key: torch.from_numpy(np.asarray(data[key])) for key in data.files}
    traj["node_type"] = traj["node_type"].long()
    traj["onehot"] = F.one_hot(traj["node_type"], NUM_NODE_TYPES).float()
    if "cells" in traj:
        traj["cells"] = traj["cells"].long()
    return traj


def predicted_mask(node_type: torch.Tensor) -> torch.Tensor:
    return torch.isin(node_type, torch.tensor(PREDICTED_NODES, device=node_type.device))


def sample_nodes(
    num_nodes: int, count: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    if count >= num_nodes:
        return torch.arange(num_nodes)
    return torch.randperm(num_nodes, generator=generator)[:count]


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories: list[dict[str, torch.Tensor]],
        num_anchors: int,
        num_queries: int,
        noise_scale: float,
        target: str,
        seed: int,
        train: bool,
    ) -> None:
        if target not in ("delta", "absolute"):
            raise ValueError(f"target must be 'delta' or 'absolute', got {target!r}")
        self.trajectories = trajectories
        self.num_anchors = num_anchors
        self.num_queries = num_queries
        self.noise_scale = noise_scale
        self.target = target
        self.seed = seed
        self.train = train
        self.windows = [
            (index, t)
            for index, traj in enumerate(trajectories)
            for t in range(traj["velocity"].shape[0] - 1)
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        traj_index, t = self.windows[index]
        traj = self.trajectories[traj_index]
        velocity, mesh_pos = traj["velocity"], traj["mesh_pos"]
        node_type, onehot = traj["node_type"], traj["onehot"]
        num_nodes = velocity.shape[1]

        current = velocity[t]
        if self.train and self.noise_scale > 0:
            noise = torch.randn_like(current) * self.noise_scale
            current = current + noise * (node_type == NORMAL_NODE).unsqueeze(-1)
        target = velocity[t + 1] if self.target == "absolute" else velocity[t + 1] - current

        if self.train:
            anchor_idx = sample_nodes(num_nodes, self.num_anchors)
            query_idx = sample_nodes(num_nodes, self.num_queries)
        else:
            generator = torch.Generator().manual_seed(self.seed + index)
            anchor_idx = sample_nodes(num_nodes, self.num_anchors, generator)
            query_idx = torch.arange(num_nodes)

        return {
            "anchor_pos": mesh_pos[anchor_idx],
            "anchor_val": torch.cat([current[anchor_idx], onehot[anchor_idx]], dim=-1),
            "query_pos": mesh_pos[query_idx],
            "query_val": onehot[query_idx],
            "query_current": current[query_idx],
            "target": target[query_idx],
            "query_mask": predicted_mask(node_type[query_idx]),
        }
