from __future__ import annotations

import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig

from dataset import load_train_dataset
from models.boosting import LightGBMTrainer


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        save_path = Path(cfg.models.path)

        train_x, train_y = load_train_dataset(cfg)

        # train model
        lgb_trainer = LightGBMTrainer(cfg)
        lgb_trainer.run_cv_training(train_x, train_y)

        # save model
        lgb_trainer.save_model(save_path / f"{cfg.models.results}.pkl")


if __name__ == "__main__":
    _main()
