import torch

from abupt.dataset import NORMAL_NODE, Dataset, predicted_mask

SAMPLING = {"num_anchors": 16, "num_queries": 24, "seed": 0}


def make_dataset(trajectories, train: bool, noise_scale: float = 0.0, target: str = "delta"):
    return Dataset(trajectories, noise_scale=noise_scale, target=target, train=train, **SAMPLING)


def test_window_count_and_shapes(trajectories):
    train_ds = make_dataset(trajectories, train=True)
    eval_ds = make_dataset(trajectories, train=False)
    expected = sum(traj["velocity"].shape[0] - 1 for traj in trajectories)
    assert len(train_ds) == len(eval_ds) == expected

    sample = train_ds[0]
    m, q = SAMPLING["num_anchors"], SAMPLING["num_queries"]
    assert sample["anchor_pos"].shape == (m, 2)
    assert sample["anchor_val"].shape == (m, 11)
    assert sample["query_pos"].shape == (q, 2)
    assert sample["query_val"].shape == (q, 9)
    assert sample["query_current"].shape == (q, 2)
    assert sample["target"].shape == (q, 2)
    assert sample["query_mask"].shape == (q,)


def test_eval_uses_full_mesh_and_no_noise(trajectories):
    sample = make_dataset(trajectories, train=False, noise_scale=10.0)[0]
    traj = trajectories[0]
    n = traj["velocity"].shape[1]
    assert sample["query_pos"].shape == (n, 2)
    torch.testing.assert_close(sample["query_pos"], traj["mesh_pos"])
    torch.testing.assert_close(sample["target"], traj["velocity"][1] - traj["velocity"][0])


def test_eval_anchor_draw_is_fixed_per_window(trajectories):
    eval_ds = make_dataset(trajectories, train=False)
    torch.testing.assert_close(eval_ds[3]["anchor_pos"], eval_ds[3]["anchor_pos"])
    assert not torch.equal(eval_ds[3]["anchor_pos"], eval_ds[4]["anchor_pos"])


def test_noise_applied_to_normal_nodes_only(trajectories):
    sample = make_dataset(trajectories, train=True, noise_scale=10.0)[0]
    traj = trajectories[0]
    anchor_idx = torch.cdist(sample["anchor_pos"], traj["mesh_pos"]).argmin(dim=1)
    raw_vel = traj["velocity"][0][anchor_idx]
    noisy_vel = sample["anchor_val"][:, :2]
    is_normal = traj["node_type"][anchor_idx] == NORMAL_NODE
    torch.testing.assert_close(noisy_vel[~is_normal], raw_vel[~is_normal])
    assert (noisy_vel[is_normal] - raw_vel[is_normal]).abs().max() > 1.0


def test_target_delta_uses_noisy_input(trajectories):
    train_ds = make_dataset(trajectories, train=True, noise_scale=0.5)
    traj_index, t = train_ds.windows[0]
    traj = trajectories[traj_index]
    torch.manual_seed(0)
    sample = train_ds[0]
    torch.manual_seed(0)
    noise = torch.randn_like(traj["velocity"][t]) * 0.5
    noisy = traj["velocity"][t] + noise * (traj["node_type"] == NORMAL_NODE).unsqueeze(-1)
    query_idx = torch.cdist(sample["query_pos"], traj["mesh_pos"]).argmin(dim=1)
    torch.testing.assert_close(
        sample["target"], traj["velocity"][t + 1][query_idx] - noisy[query_idx]
    )


def test_predicted_mask(trajectories):
    sample = make_dataset(trajectories, train=False)[0]
    node_type = trajectories[0]["node_type"]
    mask = predicted_mask(node_type)
    assert mask.dtype == torch.bool
    torch.testing.assert_close(sample["query_mask"], mask)
    assert mask[node_type == 0].all()
    assert mask[node_type == 5].all()
    assert not mask[node_type == 4].any()
    assert not mask[node_type == 6].any()


def test_absolute_target(trajectories):
    ds = make_dataset(trajectories, train=True, target="absolute")
    sample = ds[0]
    traj = trajectories[0]
    _, t = ds.windows[0]
    query_idx = torch.cdist(sample["query_pos"], traj["mesh_pos"]).argmin(dim=1)
    torch.testing.assert_close(sample["target"], traj["velocity"][t + 1][query_idx])
    torch.testing.assert_close(sample["query_current"], traj["velocity"][t][query_idx])
