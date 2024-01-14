from __future__ import annotations

from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from dataset import load_test_dataset
from omegaconf import DictConfig
from pytorch_tabnet.tab_model import TabNetRegressor
from tqdm import tqdm

from models.base import ModelResult


def inference_models(result: list[ModelResult], test_x: pd.DataFrame) -> np.ndarray:
    """
    Given a model, predict probabilities for each class.
    Args:
        model_results: ModelResult object
        test_x: test dataframe
    Returns:
        predict probabilities for each class
    """

    folds = len(result.models)
    preds = np.zeros((test_x.shape[0],))

    for model in tqdm(result.models.values(), total=folds, desc="Predicting models"):
        preds += (
            model.predict(xgb.DMatrix(test_x)) / folds
            if isinstance(model, xgb.Booster)
            else model.predict(test_x.to_numpy()) / folds
            if isinstance(model, TabNetRegressor)
            else model.predict(test_x) / folds
        )

    return preds


@hydra.main(config_path="../config/", config_name="predict")
def _main(cfg: DictConfig):
    result = joblib.load(Path(cfg.models.path) / f"{cfg.models.results}.pkl")
    test_x = load_test_dataset(cfg)

    submit = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.submit}.csv")

    preds = inference_models(result, test_x)

    submit[cfg.data.target] = preds
    submit.to_csv(Path(cfg.output.path) / f"{cfg.models.results}.csv", index=False)


if __name__ == "__main__":
    _main()
