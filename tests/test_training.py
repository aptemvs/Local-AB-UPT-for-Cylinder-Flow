import json
from pathlib import Path

import lightning as L
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from abupt.dataset import Dataset
from abupt.lit_model import load_network, masked_mse
from abupt.nn import AnchoredBranchedUPT

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_training_run(data_dir, tmp_path):
    run_dir = tmp_path / "run"
    overrides = [
        f"datamodule.data_dir={data_dir}",
        f"paths.output_dir={run_dir}",
        "datamodule.batch_size=2",
        "datamodule.num_anchors=16",
        "datamodule.num_queries=24",
        "datamodule.num_val_windows=3",
        "model.dim=32",
        "model.num_heads=2",
        "model.blocks=sl",
        "model.compile=false",
        "trainer.max_steps=4",
        "trainer.val_check_interval=2",
        "trainer.accelerator=cpu",
        "trainer.precision=32",
        "trainer.log_every_n_steps=1",
        "+trainer.enable_progress_bar=false",
        "+trainer.enable_model_summary=false",
        "callbacks.probe_rollout.num_steps=3",
        "callbacks.step_checkpoint.every_n_train_steps=2",
    ]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose("train", overrides=overrides)
    L.seed_everything(cfg.extras.seed, workers=True)
    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    logger = instantiate(cfg.logger)
    callbacks = [instantiate(callback) for callback in cfg.callbacks.values()]
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule)

    for name in ("best.ckpt", "last.ckpt", "metrics.csv", "metrics.json"):
        assert (run_dir / name).exists()
    assert sorted(p.name for p in (run_dir / "ckpts").iterdir()) == [
        "step_0000002.ckpt",
        "step_0000004.ckpt",
    ]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["num_params"] == sum(p.numel() for p in model.parameters())
    assert metrics["peak_mem_bytes"] == 0
    assert metrics["best_step"] in (2, 4)
    assert metrics["best_probe_rmse50"] > 0
    assert "validation/probe_rmse50" in (run_dir / "metrics.csv").read_text()

    network, hparams = load_network(run_dir / "ckpts" / "step_0000004.ckpt", torch.device("cpu"))
    assert hparams["blocks"] == "sl" and hparams["num_anchors"] == 16 and hparams["seed"] == 42
    torch.testing.assert_close(network.decoder.weight, model.model.decoder.weight)
    assert network.node_normalizer.acc_count.item() > 0


def test_model_memorizes_fixed_batch(trajectories):
    ds = Dataset(
        trajectories,
        num_anchors=16,
        num_queries=24,
        noise_scale=0.0,
        target="absolute",
        seed=0,
        train=True,
    )
    torch.manual_seed(0)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    model = AnchoredBranchedUPT(
        dim=32, num_heads=2, blocks="ss", local_radius=0.1, target="absolute"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    losses = []
    for _ in range(400):
        prediction = model(
            batch["anchor_pos"], batch["anchor_val"], batch["query_pos"], batch["query_val"]
        )
        target = model.normalize_target(batch["target"])
        loss = masked_mse(prediction, target, batch["query_mask"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.05
    assert losses[-1] < losses[0] / 20
