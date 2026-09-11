import numpy as np
import pytest

from abupt.dataset import load_trajectory

SPLIT_SEEDS = {"train": 0, "valid": 1, "test": 2}


def write_split(root, split: str, num_trajectories: int, frames: int = 8, nodes: int = 40) -> None:
    rng = np.random.default_rng(SPLIT_SEEDS[split])
    (root / split).mkdir(parents=True, exist_ok=True)
    for i in range(num_trajectories):
        n = nodes + i
        mesh_pos = rng.uniform((0, 0), (1.6, 0.41), size=(n, 2)).astype(np.float32)
        node_type = np.zeros(n, dtype=np.int32)
        node_type[0:3] = 4
        node_type[3:6] = 5
        node_type[6:12] = 6
        velocity = rng.normal(size=(frames, n, 2)).astype(np.float32)
        np.savez(
            root / split / f"traj_{i:04d}.npz",
            velocity=velocity,
            mesh_pos=mesh_pos,
            node_type=node_type,
            cells=np.zeros((1, 3), dtype=np.int32),
        )


@pytest.fixture()
def data_dir(tmp_path):
    write_split(tmp_path, "train", 2)
    write_split(tmp_path, "valid", 1)
    write_split(tmp_path, "test", 1)
    return tmp_path


@pytest.fixture()
def trajectories(data_dir):
    return [load_trajectory(str(path)) for path in sorted((data_dir / "train").glob("*.npz"))]
