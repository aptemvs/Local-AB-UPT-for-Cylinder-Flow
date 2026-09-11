import hashlib

import numpy as np
import torch

from abupt.dataset import predicted_mask, sample_nodes


def anchor_generator(seed: int) -> torch.Generator:
    digest = hashlib.blake2b(f"{seed}:rollout-anchor".encode(), digest_size=8).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest, "big") % 2**63)


def autocast(device: torch.device) -> torch.autocast:
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )


@torch.no_grad()
def rollout(
    model: torch.nn.Module,
    traj: dict[str, torch.Tensor],
    num_anchors: int,
    seed: int,
    max_steps: int | None = None,
) -> tuple[list[float], np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    velocity = traj["velocity"][1:].to(device)
    mesh_pos = traj["mesh_pos"].to(device)
    onehot = traj["onehot"].to(device)
    mask = predicted_mask(traj["node_type"]).to(device)

    anchor_idx = sample_nodes(mesh_pos.shape[0], num_anchors, anchor_generator(seed)).to(device)
    anchor_pos = mesh_pos[anchor_idx].unsqueeze(0)
    query_pos = mesh_pos.unsqueeze(0)
    query_val = onehot.unsqueeze(0)

    steps = velocity.shape[0] - 1
    if max_steps is not None:
        steps = min(steps, max_steps)
    state = velocity[0]
    errors: list[float] = []
    fields = [state.clone()]
    for t in range(steps):
        anchor_val = torch.cat([state[anchor_idx], onehot[anchor_idx]], dim=-1).unsqueeze(0)
        with autocast(device):
            next_state = model.predict_next(
                anchor_pos, anchor_val, query_pos, query_val, state.unsqueeze(0)
            ).squeeze(0)
        next_state = torch.where(mask.unsqueeze(-1), next_state, state)
        errors.append((next_state - velocity[t + 1]).square().mean().item())
        fields.append(next_state.clone())
        state = next_state
    return errors, torch.stack(fields).cpu().numpy()


def rollout_metrics(per_trajectory_errors: list[list[float]]) -> dict[str, float | list[float]]:
    curves = np.asarray(per_trajectory_errors)
    return {
        "rmse_1": float(np.sqrt(curves[:, 0].mean())),
        "rmse_50": float(np.sqrt(curves[:, :50].mean())),
        "rmse_all": float(np.sqrt(curves.mean())),
        "mse_per_step": curves.mean(axis=0).tolist(),
    }
