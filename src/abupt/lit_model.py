import math
from pathlib import Path

import lightning as L
import torch

from abupt.nn import AnchoredBranchedUPT


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    error = (prediction.float() - target.float()).square().sum(dim=-1)
    return error[mask].mean()


def warmup_cosine(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class LitModel(L.LightningModule):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        blocks: str,
        local_radius: float,
        pos_embed: str,
        position_scale: float,
        position_max_wavelength: float,
        target: str,
        num_anchors: int,
        seed: int,
        learning_rate: float,
        weight_decay: float,
        adam_betas: list[float],
        warmup_frac: float,
        max_steps: int,
        compile: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = AnchoredBranchedUPT(
            dim=dim,
            num_heads=num_heads,
            blocks=blocks,
            local_radius=local_radius,
            pos_embed=pos_embed,
            position_scale=position_scale,
            position_max_wavelength=position_max_wavelength,
            target=target,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_betas = tuple(adam_betas)
        self.warmup_steps = max(1, int(max_steps * warmup_frac))
        self.max_steps = max_steps
        self.training_loss = torch.compile(self.compute_loss) if compile else self.compute_loss

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(
            batch["anchor_pos"], batch["anchor_val"], batch["query_pos"], batch["query_val"]
        )

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        prediction = self(batch)
        target = self.model.normalize_target(batch["target"])
        return masked_mse(prediction, target, batch["query_mask"])

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int = 0) -> torch.Tensor:
        loss = self.training_loss(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int = 0) -> None:
        prediction = self(batch)
        target = self.model.normalize_target(batch["target"])
        mask = batch["query_mask"]
        normalizer = self.model.output_normalizer
        if self.model.absolute_target:
            persistence = batch["query_current"].float()
        else:
            persistence = torch.zeros_like(batch["target"], dtype=torch.float32)
        persistence = (persistence - normalizer.mean()) / normalizer.std_with_epsilon()
        self.log("validation/one_step_mse", masked_mse(prediction, target, mask), prog_bar=True)
        self.log("validation/persistence_mse", masked_mse(persistence, target, mask))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.adam_betas,
            weight_decay=self.weight_decay,
            fused=self.device.type == "cuda",
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: warmup_cosine(step, self.max_steps, self.warmup_steps)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


MODEL_KEYS = (
    "dim",
    "num_heads",
    "blocks",
    "local_radius",
    "pos_embed",
    "position_scale",
    "position_max_wavelength",
    "target",
)


def load_network(path: str | Path, device: torch.device) -> tuple[AnchoredBranchedUPT, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    hparams = payload.get("hyper_parameters") or payload["config"]
    network = AnchoredBranchedUPT(**{key: hparams[key] for key in MODEL_KEYS if key in hparams})
    state = payload.get("state_dict") or payload["model"]
    network.load_state_dict({key.removeprefix("model."): value for key, value in state.items()})
    return network.to(device).eval(), dict(hparams)
