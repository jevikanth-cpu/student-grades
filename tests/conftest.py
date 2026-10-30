"""Shared fixtures: a small synthetic dataset shaped like Student Performance."""

import numpy as np
import pandas as pd
import pytest

from student_grades.config import load_config


@pytest.fixture()
def config() -> dict:
    cfg = load_config()
    cfg["cv"] = {"n_splits": 3, "n_repeats": 1}
    cfg["grids"] = {
        "ridge": {"model__alpha": [1.0, 10.0]},
        "lasso": {"model__alpha": [0.1]},
        "knn": {"model__n_neighbors": [5]},
    }
    return cfg


@pytest.fixture()
def data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "age": rng.integers(15, 20, n),
            "absences": rng.integers(0, 20, n),
            "studytime": rng.integers(1, 5, n),
            "school": rng.choice(["GP", "MS"], n),
            "sex": rng.choice(["F", "M"], n),
        }
    )
    noise = rng.normal(0, 1, n)
    y = pd.Series(
        (10 + 1.5 * X["studytime"] - 0.2 * X["absences"] + noise)
        .clip(0, 20)
        .round(),
        name="G3",
    )
    return X, y
