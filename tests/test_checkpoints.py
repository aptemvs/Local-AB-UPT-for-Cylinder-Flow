import json
from pathlib import Path

import pytest
import torch

from abupt.lit_model import load_network

ROOT = Path(__file__).resolve().parents[1]
RECORDED_RUNS = {
    "ssssssss-seed42": 2_050_850,
    "lllll-r0.1-seed42": 1_298_738,
    "lllsss-r0.1-seed42": 1_549_442,
}


@pytest.mark.parametrize(("label", "num_params"), RECORDED_RUNS.items())
def test_recorded_checkpoint_loads(label, num_params):
    run_dir = ROOT / "outputs" / label
    paths = list(run_dir.glob("ckpts/step_0200000.*"))
    if not paths:
        pytest.skip(f"{label} is not available")
    network, hparams = load_network(paths[0], torch.device("cpu"))
    assert sum(p.numel() for p in network.parameters()) == num_params
    assert hparams["blocks"] == label.split("-")[0]
    assert hparams["seed"] == 42 and hparams["num_anchors"] == 600
    assert json.loads((run_dir / "metrics.json").read_text())["num_params"] == num_params
