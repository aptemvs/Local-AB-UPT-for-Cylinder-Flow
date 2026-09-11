import numpy as np

ACC_THRESHOLDS = (0.9, 0.8, 0.5)
_EPS = 1e-12


def interp_to(
    src_xy: np.ndarray, src_val: np.ndarray, dst_xy: np.ndarray, k: int = 3
) -> np.ndarray:
    d = np.linalg.norm(dst_xy[:, None, :] - src_xy[None, :, :], axis=-1)
    idx = np.argpartition(d, k, axis=1)[:, :k]
    dk = np.take_along_axis(d, idx, axis=1)
    w = 1.0 / np.maximum(dk, 1e-9)
    w /= w.sum(axis=1, keepdims=True)
    return (src_val[idx] * w[..., None]).sum(axis=1)


def moments(vel: np.ndarray) -> dict[str, np.ndarray]:
    mean = vel.mean(axis=0)
    fluct = vel - mean
    uu = (fluct[..., 0] ** 2).mean(axis=0)
    vv = (fluct[..., 1] ** 2).mean(axis=0)
    shear = (fluct[..., 0] * fluct[..., 1]).mean(axis=0)
    return {"mean": mean, "uu": uu, "vv": vv, "energy": uu + vv, "shear": shear}


def turbulence_statistics(pred_moments: dict, truth_moments: dict, u_ref: float) -> dict:
    u_ref = max(float(u_ref), _EPS)
    sides = (("pred", pred_moments), ("truth", truth_moments))
    u_rms = {k: np.sqrt(m["uu"]) for k, m in sides}
    v_rms = {k: np.sqrt(m["vv"]) for k, m in sides}
    tke = {k: 0.5 * m["energy"] for k, m in sides}
    mean_map = np.linalg.norm(pred_moments["mean"] - truth_moments["mean"], axis=-1)
    shear_error = np.abs(pred_moments["shear"] - truth_moments["shear"]).mean()
    return {
        "mae_mean_norm": float(mean_map.mean()) / u_ref,
        "mae_u_rms_norm": float(np.abs(u_rms["pred"] - u_rms["truth"]).mean()) / u_ref,
        "mae_v_rms_norm": float(np.abs(v_rms["pred"] - v_rms["truth"]).mean()) / u_ref,
        "mae_tke_norm": float(np.abs(tke["pred"] - tke["truth"]).mean()) / u_ref**2,
        "mae_shear_norm": float(shear_error) / u_ref**2,
        "tke_pred_norm": float(tke["pred"].mean()) / u_ref**2,
        "tke_truth_norm": float(tke["truth"].mean()) / u_ref**2,
        "tke_map": tke["pred"] - tke["truth"],
    }


def replay_ratio(
    pred_mean: np.ndarray, truth_mean: np.ndarray, train_means: dict[str, np.ndarray]
) -> dict:
    d_truth = float(((pred_mean - truth_mean) ** 2).mean())
    d_train = {name: float(((pred_mean - mean) ** 2).mean()) for name, mean in train_means.items()}
    nearest = min(d_train, key=d_train.get)
    return {"ratio": d_truth / max(d_train[nearest], _EPS), "nearest": nearest}


def acc_curve(pred: np.ndarray, truth: np.ndarray, climatology: np.ndarray) -> np.ndarray:
    pred_anom = (pred - climatology).reshape(len(pred), -1)
    truth_anom = (truth - climatology).reshape(len(truth), -1)
    denom = np.sqrt((pred_anom**2).sum(axis=1) * (truth_anom**2).sum(axis=1))
    return (pred_anom * truth_anom).sum(axis=1) / np.maximum(denom, _EPS)


def valid_time(
    acc: np.ndarray, dt: float, thresholds: tuple[float, ...] = ACC_THRESHOLDS, sustain: int = 5
) -> dict[float, float]:
    times = {}
    for threshold in thresholds:
        below = (acc < threshold).astype(np.float32)
        runs = np.convolve(below, np.ones(sustain), mode="valid")
        hits = np.flatnonzero(runs == sustain)
        times[threshold] = float((hits[0] if len(hits) else len(acc)) * dt)
    return times


def _triangle_geometry(mesh_pos: np.ndarray, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p0, p1, p2 = (mesh_pos[cells[:, i]] for i in range(3))
    e1, e2 = p1 - p0, p2 - p0
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    areas = 0.5 * np.abs(det)
    safe_det = np.where(np.abs(det) < _EPS, np.inf, det)
    inverse = (
        np.stack(
            [
                np.stack([e2[:, 1], -e1[:, 1]], axis=-1),
                np.stack([-e2[:, 0], e1[:, 0]], axis=-1),
            ],
            axis=1,
        )
        / safe_det[:, None, None]
    )
    return inverse, areas


def divergence_residual(fields: np.ndarray, mesh_pos: np.ndarray, cells: np.ndarray) -> dict:
    inverse, areas = _triangle_geometry(np.asarray(mesh_pos, dtype=np.float64), np.asarray(cells))
    values = np.asarray(fields, dtype=np.float64)[:, cells]
    deltas = values[:, :, 1:] - values[:, :, :1]
    gradients = np.einsum("mij,tmjc->tmic", inverse, deltas)
    divergence = gradients[..., 0, 0] + gradients[..., 1, 1]
    weights = areas / max(areas.sum(), _EPS)
    curve = np.sqrt((divergence**2 * weights).sum(axis=1))
    time_rms_map = np.sqrt((divergence**2).mean(axis=0))
    return {"curve": curve.astype(np.float32), "map": time_rms_map.astype(np.float32)}


def peak_frequency(signal: np.ndarray, dt: float) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    spectrum[0] = 0.0
    return float(np.fft.rfftfreq(len(signal), dt)[np.argmax(spectrum)])


def distribution_distance(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a, b = np.sort(np.asarray(a, dtype=np.float64)), np.sort(np.asarray(b, dtype=np.float64))
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    ks = float(np.abs(cdf_a - cdf_b).max())
    m = min(len(a), len(b))
    qa = a[np.floor(np.linspace(0, len(a) - 1, m)).astype(int)]
    qb = b[np.floor(np.linspace(0, len(b) - 1, m)).astype(int)]
    return {"ks": ks, "w1": float(np.abs(qa - qb).mean())}


def rollout_report(
    pred: np.ndarray,
    truth: np.ndarray,
    mesh_pos: np.ndarray,
    cells: np.ndarray,
    dt: float,
    start_frame: int,
    probe_xy: tuple[float, float],
    train_means: dict[str, np.ndarray],
    u_ref: float,
    diameter: float,
    stat_frame: int = 200,
) -> dict:
    mesh_pos = np.asarray(mesh_pos, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    length = min(len(pred), len(truth) - start_frame)
    pred, truth = pred[:length], truth[start_frame : start_frame + length]
    lo = max(0, stat_frame - start_frame)

    mse_per_step = ((pred[1:] - truth[1:]) ** 2).mean(axis=(1, 2))
    scalars = {
        "rmse_1": float(np.sqrt(mse_per_step[0])),
        "rmse_50": float(np.sqrt(mse_per_step[:50].mean())),
        "rmse_all": float(np.sqrt(mse_per_step.mean())),
    }
    truth_moments = moments(truth[lo:])
    pred_moments = moments(pred[lo:])
    truth_std = np.sqrt(((truth[lo:] - truth_moments["mean"]) ** 2).mean())
    for key in ("rmse_1", "rmse_50", "rmse_all"):
        scalars["v" + key] = scalars[key] / max(float(truth_std), _EPS)

    mean_error_map = np.linalg.norm(pred_moments["mean"] - truth_moments["mean"], axis=-1)
    energy_map = np.abs(pred_moments["energy"] - truth_moments["energy"])
    shear_map = np.abs(pred_moments["shear"] - truth_moments["shear"])
    scalars.update(
        mae_mean_field=float(mean_error_map.mean()),
        mae_fluct_energy=float(energy_map.mean()),
        mae_shear=float(shear_map.mean()),
        mean_flow_pred=float(np.linalg.norm(pred_moments["mean"], axis=-1).mean()),
        mean_flow_truth=float(np.linalg.norm(truth_moments["mean"], axis=-1).mean()),
        fluct_energy_pred=float(pred_moments["energy"].mean()),
        fluct_energy_truth=float(truth_moments["energy"].mean()),
    )
    turb = turbulence_statistics(pred_moments, truth_moments, u_ref)
    scalars.update({k: v for k, v in turb.items() if k != "tke_map"})

    acc = acc_curve(pred, truth, truth_moments["mean"])
    for threshold, seconds in valid_time(acc, dt).items():
        scalars[f"acc_valid_time_{int(threshold * 100)}"] = seconds

    div_pred = divergence_residual(pred, mesh_pos, cells)
    div_truth = divergence_residual(truth, mesh_pos, cells)
    scalars["div_rms_pred"] = float(np.sqrt((div_pred["curve"][lo:] ** 2).mean()))
    scalars["div_rms_truth"] = float(np.sqrt((div_truth["curve"][lo:] ** 2).mean()))
    scalars["div_ratio"] = scalars["div_rms_pred"] / max(scalars["div_rms_truth"], _EPS)

    tail = min(200, len(pred) - lo)
    scalars["amp_pred"] = float(np.linalg.norm(pred[-tail:], axis=-1).mean())
    scalars["amp_truth"] = float(np.linalg.norm(truth[-tail:], axis=-1).mean())

    probe = int(((mesh_pos - np.asarray(probe_xy, dtype=np.float32)) ** 2).sum(axis=1).argmin())
    scalars["peak_freq_pred"] = peak_frequency(pred[lo:, probe, 1], dt)
    scalars["peak_freq_truth"] = peak_frequency(truth[lo:, probe, 1], dt)
    for key in ("pred", "truth"):
        frequency = scalars[f"peak_freq_{key}"]
        scalars[f"strouhal_{key}"] = float(frequency) * float(diameter) / max(float(u_ref), _EPS)
    for channel, name in ((0, "u"), (1, "v")):
        distance = distribution_distance(pred[lo:, probe, channel], truth[lo:, probe, channel])
        scalars[f"ks_{name}"] = distance["ks"]
        scalars[f"w1_{name}"] = distance["w1"]

    replay = replay_ratio(pred_moments["mean"], truth_moments["mean"], train_means)
    scalars["replay_ratio"] = replay["ratio"]
    scalars["replay_nearest"] = replay["nearest"]

    return {
        "scalars": scalars,
        "curves": {
            "mse_per_step": mse_per_step.astype(np.float32),
            "acc": acc.astype(np.float32),
            "div_rms_pred": div_pred["curve"],
            "div_rms_truth": div_truth["curve"],
        },
        "maps": {
            "mean_field_error": mean_error_map.astype(np.float32),
            "fluct_energy_error": energy_map.astype(np.float32),
            "shear_error": shear_map.astype(np.float32),
            "div_time_rms": div_pred["map"],
            "tke_error": turb["tke_map"].astype(np.float32),
        },
    }
