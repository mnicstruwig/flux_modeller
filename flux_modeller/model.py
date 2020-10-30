"""Model module."""

from flux_modeller.postprocessing.curve import get_predictions_from_linear_model
from typing import Any, Tuple

import cloudpickle
import numpy as np


class CurveModel:
    def __init__(self,
                 regressor: Any,
                 x_arr: np.ndarray,
                 kernel_space: np.ndarray,
                 feature_engineering_pipeline: Any) -> None:
        self.regressor = regressor
        self.x_arr = x_arr
        self.kernel_space = kernel_space
        self.feature_engineering_pipeline = feature_engineering_pipeline


    def save(self, path: str) -> None:
        model_package = {
            'reg': self.regressor,
            'x_arr': self.x_arr,
            'kernel_space': self.kernel_space,
            'feature_engineering': self.feature_engineering_pipeline
        }

        with open(path, 'wb') as f:
            cloudpickle.dump(model_package, f)

    @staticmethod
    def load(path: str) -> Any:
        with open(path, 'rb') as f:
            model_package = cloudpickle.load(f)

        return CurveModel(
            model_package['reg'],
            model_package['x_arr'],
            model_package['kernel_space'],
            model_package['feature_engineering']
        )

    def predict_curve(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_transformed = self.feature_engineering_pipeline.fit_transform(X)
        y_hat = get_predictions_from_linear_model(X_transformed,
                                                  self.regressor,
                                                  self.kernel_space)

        return self.x_arr, y_hat
