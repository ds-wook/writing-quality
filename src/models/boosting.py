import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from omegaconf import DictConfig

from models.base import BaseModel


class LightGBMTrainer(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> LGBMRegressor:
        model = LGBMRegressor(**self.cfg.models.params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            callbacks=[
                lgb.log_evaluation(self.cfg.models.verbose),
                lgb.early_stopping(self.cfg.models.early_stopping_rounds),
            ],
            verbose=self.cfg.models.verbose,
        )

        return model
