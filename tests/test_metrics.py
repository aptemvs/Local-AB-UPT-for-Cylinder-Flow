import numpy as np

from abupt.metrics import (
    acc_curve,
    distribution_distance,
    divergence_residual,
    moments,
    peak_frequency,
    replay_ratio,
    rollout_report,
    turbulence_statistics,
    valid_time,
)


def grid_mesh(nx=8, ny=5):
    xs, ys = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing="ij")
    mesh_pos = np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(np.float32)
    cells = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = (i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1)
            cells += [(a, b, c), (a, c, d)]
    return mesh_pos, np.asarray(cells, dtype=np.int64)


def traveling_wave(mesh_pos, frames, period=20):
    phase = (mesh_pos[None, :, 0] - np.arange(frames)[:, None] / period) * 2 * np.pi
    return np.stack([np.sin(phase), np.cos(phase)], axis=-1).astype(np.float32)


def test_divergence_residual_matches_analytic_fields():
    mesh_pos, cells = grid_mesh()
    rotation = np.stack([-mesh_pos[:, 1], mesh_pos[:, 0]], axis=-1)[None]
    expansion = mesh_pos[None].copy()
    assert divergence_residual(rotation, mesh_pos, cells)["curve"][0] < 1e-5
    result = divergence_residual(expansion, mesh_pos, cells)
    assert abs(result["curve"][0] - 2.0) < 1e-5
    assert np.allclose(result["map"], 2.0, atol=1e-5)


def test_acc_is_one_for_perfect_and_zero_for_climatology_prediction():
    mesh_pos, _ = grid_mesh()
    truth = traveling_wave(mesh_pos, 40)
    climatology = truth.mean(axis=0)
    assert np.allclose(acc_curve(truth, truth, climatology), 1.0, atol=1e-5)
    flat = np.broadcast_to(climatology, truth.shape)
    assert np.abs(acc_curve(flat, truth, climatology)).max() < 1e-5


def test_valid_time_needs_a_sustained_crossing():
    acc = np.ones(30)
    acc[5] = 0.1
    acc[20:] = 0.1
    times = valid_time(acc, dt=0.01, thresholds=(0.8,), sustain=5)
    assert times[0.8] == 0.20
    never = valid_time(np.ones(30), dt=0.01, thresholds=(0.8,), sustain=5)
    assert never[0.8] == 0.30


def test_phase_shifted_flow_is_clean_on_stationary_metrics_but_bad_on_rmse():
    mesh_pos, _ = grid_mesh()
    truth = traveling_wave(mesh_pos, 80, period=20)
    pred = traveling_wave(mesh_pos, 85, period=20)[5:]
    pred_moments, truth_moments = moments(pred), moments(truth)
    assert np.linalg.norm(pred_moments["mean"] - truth_moments["mean"], axis=-1).mean() < 0.02
    assert np.abs(pred_moments["energy"] - truth_moments["energy"]).mean() < 0.02
    assert np.abs(pred_moments["shear"] - truth_moments["shear"]).mean() < 0.02
    distance = distribution_distance(pred[:, 0, 1], truth[:, 0, 1])
    assert distance["w1"] < 0.05
    rmse = float(np.sqrt(((pred - truth) ** 2).mean()))
    assert rmse > 0.5


def test_distribution_distance_analytic_values():
    a = np.linspace(0.0, 1.0, 101)
    same = distribution_distance(a, a)
    assert same["ks"] == 0.0 and same["w1"] == 0.0
    shifted = distribution_distance(a, a + 0.3)
    assert abs(shifted["w1"] - 0.3) < 1e-9
    assert abs(shifted["ks"] - 0.3) < 0.02
    unequal = distribution_distance(a, np.linspace(0.0, 1.0, 57))
    assert unequal["ks"] < 0.02 and unequal["w1"] < 0.02


def test_peak_frequency_recovers_a_sinusoid_within_one_bin():
    t = np.arange(800) * 0.01
    assert abs(peak_frequency(np.sin(2 * np.pi * 4.4 * t), 0.01) - 4.4) < 1 / 8.0


def test_replay_ratio_flags_a_parked_rollout():
    truth_mean = np.zeros((10, 2))
    train_means = {"a": np.full((10, 2), 1.0), "b": np.full((10, 2), 3.0)}
    parked = replay_ratio(train_means["a"], truth_mean, train_means)
    assert parked["nearest"] == "a" and parked["ratio"] > 1e6
    genuine = replay_ratio(truth_mean + 0.01, truth_mean, train_means)
    assert genuine["ratio"] < 1.0


def test_moments_are_time_mean_and_variance_of_a_sinusoid():
    t = np.arange(2000) * 0.01
    u = 1.5 + 2.0 * np.sin(2 * np.pi * 4.0 * t)
    v = 2.0 * np.cos(2 * np.pi * 4.0 * t)
    vel = np.stack([u, v], axis=-1)[:, None, :]
    m = moments(vel)
    assert abs(m["mean"][0, 0] - 1.5) < 1e-3
    assert abs(m["uu"][0] - np.var(u)) < 1e-9
    assert abs(m["uu"][0] - 2.0) < 1e-2
    assert abs(m["shear"][0] - np.cov(u, v, bias=True)[0, 1]) < 1e-9
    assert abs(m["shear"][0]) < 1e-2
    turb = turbulence_statistics(m, m, u_ref=2.0)
    assert abs(turb["tke_truth_norm"] - 0.5) < 1e-2
    assert turb["mae_u_rms_norm"] == 0.0 and turb["mae_tke_norm"] == 0.0
    assert abs(np.linalg.norm(m["mean"], axis=-1).mean() - 1.5) < 1e-3
    assert abs(m["energy"].mean() - 4.0) < 1e-2


def test_turbulence_statistics_normalization_is_scale_invariant():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(200, 6, 2))
    pred = truth + rng.normal(scale=0.1, size=truth.shape)
    base = turbulence_statistics(moments(pred), moments(truth), u_ref=2.0)
    scaled = turbulence_statistics(moments(pred * 3), moments(truth * 3), u_ref=6.0)
    for key in (
        "mae_mean_norm",
        "mae_u_rms_norm",
        "mae_v_rms_norm",
        "mae_tke_norm",
        "mae_shear_norm",
    ):
        assert abs(base[key] - scaled[key]) < 1e-12


def test_rollout_report_on_a_perfect_warm_prediction():
    mesh_pos, cells = grid_mesh()
    truth = traveling_wave(mesh_pos, 120)
    report = rollout_report(
        truth[40:],
        truth,
        mesh_pos,
        cells,
        dt=0.01,
        start_frame=40,
        stat_frame=40,
        probe_xy=(0.5, 0.5),
        train_means={"other": truth[40:].mean(axis=0) + 1.0},
        u_ref=2.0,
        diameter=0.1,
    )
    scalars = report["scalars"]
    assert scalars["rmse_all"] < 1e-6 and scalars["vrmse_all"] < 1e-6
    assert scalars["mae_mean_field"] < 1e-6
    assert scalars["acc_valid_time_80"] == 0.80
    assert abs(scalars["div_ratio"] - 1.0) < 1e-6
    assert scalars["ks_u"] == 0.0 and scalars["w1_v"] == 0.0
    assert scalars["mean_flow_pred"] == scalars["mean_flow_truth"]
    assert scalars["fluct_energy_pred"] == scalars["fluct_energy_truth"]
    assert abs(scalars["peak_freq_pred"] - scalars["peak_freq_truth"]) < 1e-9
    assert scalars["strouhal_pred"] == scalars["strouhal_truth"]
    assert abs(scalars["strouhal_pred"] - scalars["peak_freq_pred"] * 0.1 / 2.0) < 1e-12
    assert scalars["replay_nearest"] == "other" and scalars["replay_ratio"] < 1e-6
    assert report["maps"]["tke_error"].shape == (len(mesh_pos),)
    assert report["curves"]["acc"].shape == (80,)
    assert report["maps"]["mean_field_error"].shape == (len(mesh_pos),)
    assert report["maps"]["div_time_rms"].shape == (len(cells),)
