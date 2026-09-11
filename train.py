import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision(cfg.extras.float32_matmul_precision)
    L.seed_everything(cfg.extras.seed, workers=True)

    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    logger = instantiate(cfg.logger)
    callbacks = [instantiate(callback) for callback in cfg.callbacks.values()]
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
