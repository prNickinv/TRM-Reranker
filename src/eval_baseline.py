"""Evaluation script for Baseline Pretrained Models."""

import warnings
warnings.filterwarnings("ignore")

import hydra
import lightning
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from omegaconf import DictConfig

from utils.logging_utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiate loggers from config."""
    loggers = []
    if logger_cfg:
        for _, lg_conf in logger_cfg.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers


@hydra.main(version_base="1.3", config_path="../configs", config_name="experiment/eval_baseline.yaml")
def main(cfg: DictConfig):
    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))
    
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    # For evaluation, we usually only need a basic trainer
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers)
    
    log.info("Starting Validation on Pretrained Baseline!")
    trainer.test(model=model, datamodule=datamodule)
    #if trainer.logger:
    #    trainer.logger.finalize('success')
    
    log.info("Evaluation finished!")

if __name__ == "__main__":
    main()
_