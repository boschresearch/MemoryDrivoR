# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import glob
import logging
import os
import pickle
from datetime import datetime

import hydra
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.script.run_training import build_datasets, dist_ready
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "b2d_training"


def find_latest_checkpoint(search_pattern):
    list_of_files = glob.glob(search_pattern, recursive=True)
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getmtime)



@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    logger.info(f"Global Seed set to {cfg.seed}")
    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(agent=agent)
    lightning_module.strict_loading = cfg.strict_load

    if cfg.use_cache_without_dataset:
        logger.info("Using cached B2D data without building SceneLoader")
        assert not cfg.force_cache_computation, "force_cache_computation must be False when using cache-only B2D data"
        assert cfg.cache_path is not None, "cache_path must be provided for B2D cache-only training"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True, drop_last=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False, drop_last=True)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    if cfg.train_ckpt_path is None:
        search_pattern = "/".join(str(cfg.output_dir).split("/")[:-1]) + "/*/lightning_logs/version_*/checkpoints/" + "*.ckpt"
        cfg.train_ckpt_path = find_latest_checkpoint(search_pattern)
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())

    if cfg.validation_run:
        logger.info("Starting Validation")
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        dump_root = os.path.join(os.getenv("SUBSCORE_PATH"), "navsim1_pdm_scores", cfg.experiment_name)
        os.makedirs(dump_root, exist_ok=True)
        dump_path = os.path.join(dump_root, f"{timestamp}.pkl")
        trainer.validate(
            model=lightning_module,
            dataloaders=[val_dataloader],
            ckpt_path=cfg.train_ckpt_path,
            verbose=True,
        )
        logger.info("Running predictions to collect trajectories")
        predictions = trainer.predict(
            AgentLightningModule(agent=agent, for_viz=True),
            val_dataloader,
            return_predictions=True,
        )

        if dist_ready():
            dist.barrier()

        world_size = dist.get_world_size() if dist_ready() else 1
        all_predictions = [None for _ in range(world_size)]

        if dist_ready():
            dist.all_gather_object(all_predictions, predictions)
        else:
            all_predictions = [predictions]

        rank = dist.get_rank() if dist_ready() else 0
        if rank != 0:
            return None

        for proc_prediction in all_predictions:
            del proc_prediction

        pickle.dump(predictions, open(dump_path, "wb"))
    else:
        logger.info("Starting Training")
        fit_ckpt_path = cfg.train_ckpt_path
        load_training_state = cfg.get("load_state", cfg.strict_load)

        if cfg.train_ckpt_path is not None and not load_training_state:
            logger.info(
                "Warm-starting model weights only from checkpoint (strict=%s): %s",
                cfg.strict_load,
                cfg.train_ckpt_path,
            )
            checkpoint = torch.load(cfg.train_ckpt_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            lightning_module.load_state_dict(state_dict, strict=cfg.strict_load)
            fit_ckpt_path = None
        elif cfg.train_ckpt_path is not None:
            logger.info("Resuming full training state from checkpoint: %s", cfg.train_ckpt_path)

        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=fit_ckpt_path,
        )


if __name__ == "__main__":
    main()
