import pytest
import torch

from abupt.dataset import load_trajectory, predicted_mask
from abupt.lit_model import masked_mse, warmup_cosine
from abupt.rollout import anchor_generator, rollout, rollout_metrics


class ConstantDeltaModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    @torch.no_grad()
    def predict_next(self, anchor_pos, anchor_val, query_pos, query_onehot, current):
        return current + 1.0


def test_rollout_holds_boundary_nodes_fixed(data_dir):
    traj = load_trajectory(str(data_dir / "valid" / "traj_0000.npz"))
    traj["velocity"] = traj["velocity"][:4].clone()
    traj["velocity"][:] = traj["velocity"][0]
    errors, fields = rollout(ConstantDeltaModel(), traj, num_anchors=16, seed=0)
    fraction = predicted_mask(traj["node_type"]).float().mean().item()
    assert len(errors) == 2
    assert errors[0] == pytest.approx(fraction, rel=1e-5)
    assert errors[1] == pytest.approx(4 * fraction, rel=1e-5)
    assert fields.shape == (3, traj["velocity"].shape[1], 2)


def test_rollout_respects_max_steps(data_dir):
    traj = load_trajectory(str(data_dir / "valid" / "traj_0000.npz"))
    errors, fields = rollout(ConstantDeltaModel(), traj, num_anchors=16, seed=0, max_steps=3)
    assert len(errors) == 3
    assert fields.shape == (4, traj["velocity"].shape[1], 2)


def test_anchor_generator_is_frozen():
    assert anchor_generator(42).initial_seed() == 4353665861661328055
    assert anchor_generator(1000).initial_seed() == 6596020024732183027
    first = torch.randperm(100, generator=anchor_generator(42))
    torch.testing.assert_close(first, torch.randperm(100, generator=anchor_generator(42)))
    assert not torch.equal(first, torch.randperm(100, generator=anchor_generator(43)))


def test_rollout_metrics_aggregation():
    metrics = rollout_metrics([[1.0, 4.0, 9.0], [1.0, 4.0, 9.0]])
    assert metrics["rmse_1"] == pytest.approx(1.0)
    assert metrics["rmse_all"] == pytest.approx((14 / 3) ** 0.5)


def test_masked_mse_ignores_masked_out_nodes():
    prediction = torch.zeros(1, 4, 2)
    target = torch.ones(1, 4, 2)
    mask = torch.tensor([[True, True, False, False]])
    assert masked_mse(prediction, target, mask).item() == pytest.approx(2.0)
    target[0, 2:] = 1000.0
    assert masked_mse(prediction, target, mask).item() == pytest.approx(2.0)


def test_warmup_cosine_shape():
    assert warmup_cosine(0, 100, 10) == pytest.approx(0.1)
    assert warmup_cosine(9, 100, 10) == pytest.approx(1.0)
    assert warmup_cosine(11, 100, 10) < 1.0
    assert warmup_cosine(99, 100, 10) == pytest.approx(0.0, abs=1e-3)
