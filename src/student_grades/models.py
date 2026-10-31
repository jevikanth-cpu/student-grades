"""Training: pipelines, cross-validation, model comparison, save and load."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from student_grades.evaluate import regression_metrics
from student_grades.features import build_preprocessor


class Trainer:
    """Build pipelines, compare models under the same CV, save and load.

    Args:
        config: Dictionary returned by :func:`student_grades.config.load_config`.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        cv = config["cv"]
        self.cv = RepeatedKFold(
            n_splits=cv["n_splits"],
            n_repeats=cv["n_repeats"],
            random_state=config["seed"],
        )
        self.scoring: str = config["scoring"]
        self.best_name: str | None = None
        self.best_model: Pipeline | None = None

    def build_pipeline(self, estimator: Any) -> Pipeline:
        """Chain the preprocessor and an estimator (step names: prep, model)."""
        return Pipeline([("prep", build_preprocessor()), ("model", estimator)])

    def candidates(self) -> dict[str, tuple[Any, dict[str, list]]]:
        """Models to compare, with their hyperparameter grids from the config."""
        grids = self.config["grids"]
        seed = self.config["seed"]
        return {
            "ridge": (Ridge(), grids["ridge"]),
            "lasso": (Lasso(max_iter=10_000, random_state=seed), grids["lasso"]),
            "knn": (KNeighborsRegressor(), grids["knn"]),
            "huber": (HuberRegressor(max_iter=1000), grids["huber"]),
        }

    def cv_score(
        self, estimator: Any, X: pd.DataFrame, y: pd.Series
    ) -> tuple[float, float]:
        """Cross-validated RMSE (mean, std) of an estimator inside the pipeline."""
        scores = cross_val_score(
            self.build_pipeline(estimator), X, y, cv=self.cv, scoring=self.scoring
        )
        return float(-scores.mean()), float(scores.std())

    def baselines(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """CV RMSE of the dummy baseline and of plain linear regression."""
        rows = []
        for name, est in [
            ("dummy_mean", DummyRegressor(strategy="mean")),
            ("linear_regression", LinearRegression()),
        ]:
            mean, std = self.cv_score(est, X, y)
            rows.append({"model": name, "cv_rmse": mean, "cv_std": std})
        return pd.DataFrame(rows)

    def compare(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """Tune each candidate with GridSearchCV on the training set only.

        Returns:
            One row per model, sorted by CV RMSE. The best fitted pipeline is
            stored in ``self.best_model``.
        """
        rows, fitted = [], {}
        for name, (est, grid) in self.candidates().items():
            search = GridSearchCV(
                self.build_pipeline(est), grid, cv=self.cv, scoring=self.scoring
            )
            search.fit(X_train, y_train)
            fitted[name] = search
            rows.append(
                {
                    "model": name,
                    "cv_rmse": float(-search.best_score_),
                    "best_params": search.best_params_,
                }
            )
        table = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
        self.best_name = table.loc[0, "model"]
        self.best_model = fitted[self.best_name].best_estimator_
        return table

    def test_score(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Score the chosen model on the test set. Call this ONCE, at the end."""
        return regression_metrics(y_test, self.predict(X_test))

    def predict(self, X: pd.DataFrame) -> Any:
        """Predict grades, clipped to the valid 0-20 range."""
        if self.best_model is None:
            raise RuntimeError("No trained model: call compare() or load() first.")
        return self.best_model.predict(X).clip(0, 20)

    def save(self, path: Path) -> None:
        """Save the fitted pipeline with joblib."""
        if self.best_model is None:
            raise RuntimeError("Nothing to save: train a model first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, path)

    def load(self, path: Path) -> None:
        """Load a pipeline saved with :meth:`save`."""
        self.best_model = joblib.load(path)
