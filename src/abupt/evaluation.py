from collections.abc import Iterable

import numpy as np
import torch

from abupt.dataset import predicted_mask, sample_nodes
from abupt.metrics import interp_to, moments, peak_frequency, rollout_report
from abupt.rollout import anchor_generator, autocast, rollout

DT = 0.01
DIAMETER = 0.1
PROBE_OFFSET = 0.34
STAT_FRAME = 200
LAST_FRAME = 1000

Trajectory = dict[str, torch.Tensor]
MeanReference = dict[str, tuple[np.ndarray, np.ndarray]]


def probe_index(traj: Trajectory) -> int:
    mesh_pos = traj["mesh_pos"].numpy()
    return int(
        np.argmin(
            np.hypot(
                mesh_pos[:, 0] - (float(traj["cx"]) + PROBE_OFFSET),
                mesh_pos[:, 1] - float(traj["cy"]),
            )
        )
    )


def training_reference(trajectories: dict[str, Trajectory]) -> tuple[MeanReference, float]:
    means: MeanReference = {}
    max_speed = 0.0
    for sim, traj in trajectories.items():
        velocity = traj["velocity"][STAT_FRAME:].numpy()
        means[sim] = (traj["mesh_pos"].numpy(), velocity.mean(axis=0))
        max_speed = max(max_speed, float(np.linalg.norm(velocity, axis=-1).max()))
    return means, max_speed


def train_means_on(mesh_pos: np.ndarray, reference: MeanReference) -> dict[str, np.ndarray]:
    return {
        name: value if np.array_equal(xy, mesh_pos) else interp_to(xy, value, mesh_pos)
        for name, (xy, value) in reference.items()
    }


def truth_stats(sim: str, traj: Trajectory, train_max_speed: float) -> dict[str, float | int | str]:
    velocity = traj["velocity"][STAT_FRAME:].numpy()
    speed = np.linalg.norm(velocity, axis=-1)
    m = moments(velocity)
    umax = float(traj["umax"])
    frequency = peak_frequency(velocity[:, probe_index(traj), 1], DT)
    return {
        "sim": sim,
        "umax": umax,
        "cx": float(traj["cx"]),
        "cy": float(traj["cy"]),
        "nodes": int(velocity.shape[1]),
        "max_speed": float(speed.max()),
        "p99_speed": float(np.percentile(speed, 99)),
        "mean_speed": float(speed.mean()),
        "frac_above_train_max": float((speed > train_max_speed).mean()),
        "mean_flow": float(np.linalg.norm(m["mean"], axis=-1).mean()),
        "fluct_energy": float(m["energy"].mean()),
        "peak_freq": float(frequency),
        "strouhal": float(frequency * DIAMETER / umax),
    }


@torch.no_grad()
def one_step_rmse(
    model: torch.nn.Module,
    traj: Trajectory,
    num_anchors: int,
    seed: int,
    frames: Iterable[int] = range(STAT_FRAME, LAST_FRAME),
    chunk: int = 16,
) -> float:
    device = next(model.parameters()).device
    velocity = traj["velocity"].to(device)
    mesh_pos = traj["mesh_pos"].to(device)
    onehot = traj["onehot"].to(device)
    mask = predicted_mask(traj["node_type"]).to(device)
    anchor_idx = sample_nodes(mesh_pos.shape[0], num_anchors, anchor_generator(seed)).to(device)
    frames = list(frames)
    squared_error, count = 0.0, 0
    for i in range(0, len(frames), chunk):
        ts = frames[i : i + chunk]
        batch = len(ts)
        current = velocity[ts]
        anchor_val = torch.cat(
            [current[:, anchor_idx], onehot[anchor_idx].expand(batch, -1, -1)], dim=-1
        )
        with autocast(device):
            prediction = model.predict_next(
                mesh_pos[anchor_idx].expand(batch, -1, -1),
                anchor_val,
                mesh_pos.expand(batch, -1, -1),
                onehot.expand(batch, -1, -1),
                current,
            )
        target = velocity[[t + 1 for t in ts]]
        prediction = torch.where(mask[None, :, None], prediction, target)
        squared_error += float((prediction - target).square().sum())
        count += target.numel()
    return float(np.sqrt(squared_error / count))


def rollout_scalars(
    model: torch.nn.Module,
    traj: Trajectory,
    num_anchors: int,
    seed: int,
    start: int,
    train_means: dict[str, np.ndarray],
    train_max_speed: float,
) -> tuple[dict, np.ndarray]:
    warm = {**traj, "velocity": traj["velocity"][start - 1 :]}
    _, fields = rollout(model, warm, num_anchors, seed)
    fields = fields.astype(np.float32)
    truth = traj["velocity"].numpy()
    umax = float(traj["umax"])
    report = rollout_report(
        fields,
        truth,
        traj["mesh_pos"].numpy(),
        traj["cells"].numpy(),
        dt=DT,
        start_frame=start,
        stat_frame=STAT_FRAME,
        probe_xy=(float(traj["cx"]) + PROBE_OFFSET, float(traj["cy"])),
        train_means=train_means,
        u_ref=umax,
        diameter=DIAMETER,
    )
    scalars = dict(report["scalars"])
    pred, ref = fields[1:], truth[start + 1 : start + fields.shape[0]]
    scalars["mae_time_mean_norm"] = float(np.abs(pred.mean(0) - ref.mean(0)).mean() / umax)
    scalars["mae_time_std_norm"] = float(np.abs(pred.std(0) - ref.std(0)).mean() / umax)
    scalars["mae_time_mean"] = float(np.abs(pred.mean(0) - ref.mean(0)).mean())
    scalars["mae_time_std"] = float(np.abs(pred.std(0) - ref.std(0)).mean())
    speed = np.linalg.norm(pred, axis=-1)
    scalars["pred_max_speed"] = float(speed.max())
    scalars["pred_p99_speed"] = float(np.percentile(speed, 99))
    scalars["pred_frac_above_train_max"] = float((speed > train_max_speed).mean())
    scalars["train_max_speed"] = train_max_speed
    scalars.update(start=start, umax=umax, frames=int(fields.shape[0] - 1))
    return scalars, fields
