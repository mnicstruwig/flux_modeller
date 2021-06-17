"""Model module."""

from typing import Any, Tuple

import cloudpickle
import numpy as np

# TODO: Write method docstrings.
# TODO: Use this class to train a curve model as well.
class CurveModel:
    def __init__(self,
                 regressor: Any,
                 x_arr: np.ndarray,
                 kernel_space: np.ndarray,
                 feature_engineering_pipeline: Any) -> None:
        """Create a curve model that approximates a flux curve.

        Note, this is currently used to package the various trained model
        components, and not to train a curve model.

        Parameters
        ----------
        regressor: any
            A trained curve regressor that supports the sklearn model API.
        x_arr : np.ndarray
            Position of the lowermost magnet in the magnet assembly, relative to
            the top of the fixed magnet. Expected to correspond to the points
            produced by the model. This value is not used for any computation.
        kernel_space: np.ndarray
            (p, k) array representing the kernel space. Where `p` is the number
            of points in `x_arr` and `k` is the number of kernels.
        feature_engineering_pipeline: any
            An object that supports the sklearn pipeline API. Called to
            transform input features `X` during prediction.

        """
        self.regressor = regressor
        self.x_arr = x_arr if isinstance(x_arr, np.ndarray) else np.array(x_arr)
        self.kernel_space = kernel_space
        self.feature_engineering_pipeline = feature_engineering_pipeline
    def __repr__(self):
        return f'CurveModel(regressor={self.regressor}, x_arr.shape={self.x_arr.shape}, kernel_space.shape={self.kernel_space.shape}, feature_engineering_pipeline={self.feature_engineering_pipeline})'  # noqa

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
            regressor=model_package['reg'],
            x_arr=model_package['x_arr'],
            kernel_space=model_package['kernel_space'],
            feature_engineering_pipeline=model_package['feature_engineering']
        )

    def predict_curves(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_transformed = self.feature_engineering_pipeline.fit_transform(X)
        y_hat = self.get_predictions_from_linear_model(X_transformed,
                                                       self.regressor,
                                                       self.kernel_space)

        return self.x_arr, y_hat


    @staticmethod
    def get_predictions_from_linear_model(X: np.ndarray,
                                          reg: Any,
                                          kernel_space: np.ndarray) -> np.ndarray:
        """Calculate the predicted flux curves.

        Parameters
        ----------
        X : array(n, d)[float]
            The input training data consisting of `n` samples and `d` dimensions.
        reg : LinearRegression
            The fitted LinearRegression model that predicts the kernel weights from `X`
        kernel_space : array(p, k)[float]
            The kernel space

        Returns
        -------
        array(n, p)[float]
            The predicted flux curves.

        """
        kernel_weights_hat = reg.predict(X)
        y_hat = kernel_space.dot(kernel_weights_hat.T)
        return np.array(y_hat).T
