"""Evaluation script for TRM Reranker."""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from lightning import Trainer

from utils.logging_utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="evaluate.yaml")
def main(cfg: DictConfig):
    """Main evaluation function."""
    
    log.info(f"Loading model from checkpoint: {cfg.checkpoint}")
    
    # Import model class
    from models.trm_reranker import TRMReranker
    model = TRMReranker.load_from_checkpoint(cfg.checkpoint)
    model.eval()
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    
    log.info("Instantiating trainer...")
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
    )
    
    log.info("Starting evaluation...")
    results = trainer.test(model=model, datamodule=datamodule)
    
    log.info("Evaluation Results:")
    for key, value in results[0].items():
        log.info(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
