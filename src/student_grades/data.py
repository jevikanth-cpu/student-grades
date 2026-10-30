"""Data module: download, clean and split the dataset."""

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetLoader:
    """Download, clean and split the Student Performance dataset.

    Args:
        config: Dictionary returned by :func:`student_grades.config.load_config`.
    """

    PREVIOUS_GRADES = ("G1", "G2")

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.target: str = config["target"]
        self.seed: int = config["seed"]

    def load_raw(self) -> tuple[pd.DataFrame, pd.Series]:
        """Download the dataset from the UCI repository.

        Returns:
            Features ``X`` and target ``y`` (final grade). G1/G2 are kept in ``X``
            only if ``use_previous_grades`` is true in the config.
        """
        from ucimlrepo import fetch_ucirepo  # imported here: tests need no network

        data = fetch_ucirepo(id=self.config["uci_id"])
        X = data.data.features.copy()
        targets = data.data.targets
        if self.config["use_previous_grades"]:
            for col in self.PREVIOUS_GRADES:
                X[col] = targets[col]
        return X, targets[self.target].copy()

    def clean(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Drop duplicates, rows without target, and enforce the grade range.

        Args:
            X: Raw features.
            y: Raw target.

        Returns:
            Cleaned ``(X, y)`` with a fresh index.
        """
        df = X.copy()
        df[self.target] = y
        df = df.dropna(subset=[self.target]).drop_duplicates()
        df = df[df[self.target].between(0, 20)]
        df = df.reset_index(drop=True)
        return df.drop(columns=[self.target]), df[self.target]

    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split once into train and test, stratified on target quartiles.

        Args:
            X: Cleaned features.
            y: Cleaned target.

        Returns:
            ``X_train, X_test, y_train, y_test``.
        """
        strata = pd.qcut(y, q=4, labels=False, duplicates="drop")
        return train_test_split(
            X,
            y,
            test_size=self.config["test_size"],
            random_state=self.seed,
            stratify=strata,
        )

    def get_splits(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Run download, cleaning and split in one call."""
        X, y = self.load_raw()
        X, y = self.clean(X, y)
        return self.split(X, y)
