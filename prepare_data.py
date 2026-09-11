import argparse
import csv
import json
from pathlib import Path

import meshio
import numpy as np

NODE_NORMAL = 0
NODE_INFLOW = 4
NODE_OUTFLOW = 5
NODE_WALL = 6
TRAIN_SIMS = tuple(f"{r}_u{u}" for r in ("r4", "r5", "r6") for u in ("2.0", "2.1", "2.2", "2.5"))
EXPERIMENTS = {
    "exp00": (("r5_u2.0",), "r5_u2.0", "r5_u2.0"),
    "exp15": (("r5_u2.0",), "r6_u2.0", "r8_u2.0"),
    "exp40": (TRAIN_SIMS, "r5_u2.4", "r5_u2.3"),
}
SPEEDS = (
    "2.0",
    "2.1",
    "2.2",
    "2.3",
    "2.4",
    "2.5",
    "2.6",
    "2.7",
    "2.8",
    "2.9",
    "3.0",
    "3.1",
    "3.2",
    "3.3",
    "3.6",
    "4.0",
)
EVALUATION_SIMS = (
    *(f"r{i}_u2.2" for i in range(1, 16)),
    *(f"r5_u{u}" for u in SPEEDS),
    "r4_u2.3",
    "r6_u2.4",
    "r5_u2.0_meshB",
    "r5_u2.0_meshC",
)
EXPECTED_FRAMES = 1001
FRAME_DT = 0.01
BOUNDARY_TOL = 1e-6
CYLINDER_TOL = 1e-4


def classify_nodes(xy: np.ndarray, meta: dict) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    node_type = np.full(len(xy), NODE_NORMAL, dtype=np.int32)
    node_type[x < BOUNDARY_TOL] = NODE_INFLOW
    node_type[x > meta["L"] - BOUNDARY_TOL] = NODE_OUTFLOW
    wall = (
        (y < BOUNDARY_TOL)
        | (y > meta["H"] - BOUNDARY_TOL)
        | (np.hypot(x - meta["cx"], y - meta["cy"]) <= meta["radius"] + CYLINDER_TOL)
    )
    node_type[wall] = NODE_WALL
    return node_type


def edge_stats(cells: np.ndarray, xy: np.ndarray) -> dict[str, float]:
    edges = np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1)
    return {
        "min": float(lengths.min()),
        "p1": float(np.percentile(lengths, 1)),
        "median": float(np.median(lengths)),
    }


def convert_sim(vtu_path: Path, meta: dict, out_path: Path) -> dict:
    mesh = meshio.read(vtu_path)
    xy64 = mesh.points[:, :2].astype(np.float64)
    cells = mesh.cells_dict["triangle"].astype(np.int32)

    series: dict[str, dict[float, np.ndarray]] = {"u": {}, "v": {}, "p": {}}
    for name, arr in mesh.point_data.items():
        if "_@_t=" not in name:
            continue
        base, tstr = name.split("_@_t=")
        t = float(tstr)
        if base.endswith("x-component"):
            series["u"][t] = arr
        elif base.endswith("y-component"):
            series["v"][t] = arr
        elif base.startswith("Pressure"):
            series["p"][t] = arr

    times = sorted(series["u"])
    assert len(times) == EXPECTED_FRAMES, f"{vtu_path.name}: {len(times)} frames"
    assert sorted(series["v"]) == times and sorted(series["p"]) == times
    assert np.allclose(np.diff(times), FRAME_DT, atol=1e-9), "output grid is not exact 0.01 s"

    velocity = np.stack(
        [np.stack([series["u"][t], series["v"][t]], axis=-1) for t in times]
    ).astype(np.float32)
    pressure = np.stack([series["p"][t] for t in times]).astype(np.float32)
    node_type = classify_nodes(xy64, meta)

    num_nodes = xy64.shape[0]
    assert np.isfinite(velocity).all() and np.isfinite(pressure).all(), "NaN/Inf in fields"
    assert int(meta["n_vertices"]) == num_nodes, "manifest/vtu vertex mismatch"
    assert int(meta["n_triangles"]) == cells.shape[0], "manifest/vtu triangle mismatch"
    assert xy64[:, 0].min() >= -1e-9 and xy64[:, 0].max() <= meta["L"] + 1e-9
    assert xy64[:, 1].min() >= -1e-9 and xy64[:, 1].max() <= meta["H"] + 1e-9
    counts = {
        kind: int((node_type == kind).sum())
        for kind in (NODE_NORMAL, NODE_INFLOW, NODE_OUTFLOW, NODE_WALL)
    }
    assert all(counts[k] > 0 for k in (NODE_INFLOW, NODE_OUTFLOW, NODE_WALL)), counts
    start_max = float(np.abs(velocity[0]).max())
    final_max = float(np.abs(velocity[-1]).max())
    assert start_max < 1e-6, f"ramped start should be ~0 at t=0, got {start_max}"
    assert 1.0 < final_max < 15.0, f"suspicious final-frame velocity {final_max}"

    np.savez(
        out_path,
        velocity=velocity,
        mesh_pos=xy64.astype(np.float32),
        node_type=node_type,
        cells=cells,
        pressure=pressure,
        umax=np.float32(meta["Umax"]),
        cx=np.float32(meta["cx"]),
        cy=np.float32(meta["cy"]),
    )
    return {
        "nodes": num_nodes,
        "triangles": int(cells.shape[0]),
        "type_counts": counts,
        "final_max_vel": final_max,
        "edges": edge_stats(cells, xy64),
    }


def read_manifest(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8-sig") as fp:
        return {
            row["sim_id"]: {
                key: value if key in ("sim_id", "mesh_variant") else float(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(fp)
        }


def convert_missing(src: Path, dst: Path, missing: list[str]) -> None:
    absent = [sim for sim in missing if not (src / f"{sim}.vtu").exists()]
    assert not absent, f"missing VTU exports: {absent}"
    (dst / "manifest.csv").write_bytes((src / "manifest.csv").read_bytes())
    manifest = read_manifest(dst / "manifest.csv")
    for sim in missing:
        info = convert_sim(src / f"{sim}.vtu", manifest[sim], dst / "pool" / f"{sim}.npz")
        edges = info["edges"]
        print(
            f"{sim}: {info['nodes']} nodes / {info['triangles']} tris, "
            f"types {info['type_counts']}, |u|_final {info['final_max_vel']:.2f}, "
            f"edge min/p1/med {edges['min'] * 1e3:.2f}/{edges['p1'] * 1e3:.2f}/"
            f"{edges['median'] * 1e3:.2f} mm",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the npz pool and the experiment split trees"
    )
    parser.add_argument("--src", type=Path, default=None)
    parser.add_argument("--dst", type=Path, default=Path("data/comsol_cylinder"))
    args = parser.parse_args()
    pool = args.dst / "pool"
    pool.mkdir(parents=True, exist_ok=True)

    needed = sorted(
        {sim for train, valid, test in EXPERIMENTS.values() for sim in (*train, valid, test)}
        | set(EVALUATION_SIMS)
    )
    missing = [sim for sim in needed if not (pool / f"{sim}.npz").exists()]
    if missing:
        assert args.src is not None, f"pool lacks {missing}; pass --src to convert the VTU exports"
        convert_missing(args.src, args.dst, missing)

    for exp, (train, valid, test) in EXPERIMENTS.items():
        for split, sims in (("train", train), ("valid", (valid,)), ("test", (test,))):
            split_dir = args.dst / exp / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for i, sim in enumerate(sims):
                link = split_dir / f"traj_{i:04d}.npz"
                link.unlink(missing_ok=True)
                link.symlink_to(Path("..") / ".." / "pool" / f"{sim}.npz")
    (args.dst / "experiments.json").write_text(json.dumps(EXPERIMENTS, indent=2))
    print(
        f"pool: {len(needed)} sims ({len(missing)} converted); "
        f"{len(EXPERIMENTS)} experiment trees under {args.dst}/"
    )


if __name__ == "__main__":
    main()
