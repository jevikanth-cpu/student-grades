import numpy as np
from sklearn.dummy import DummyRegressor

from student_grades.data import DatasetLoader
from student_grades.features import build_preprocessor
from student_grades.models import Trainer


def test_scaler_is_fitted_on_train_only(config, data):
    X_tr, X_te, _, _ = DatasetLoader(config).split(*data)
    prep = build_preprocessor().fit(X_tr)
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    num_cols = ["age", "absences", "studytime"]
    np.testing.assert_allclose(scaler.mean_, X_tr[num_cols].mean().to_numpy())
    assert not np.allclose(scaler.mean_, data[0][num_cols].mean().to_numpy())


def test_pipeline_handles_unseen_category(config, data):
    X_tr, X_te, y_tr, _ = DatasetLoader(config).split(*data)
    pipe = Trainer(config).build_pipeline(DummyRegressor()).fit(X_tr, y_tr)
    X_new = X_te.head(3).copy()
    X_new["school"] = "NEW_SCHOOL"
    assert pipe.predict(X_new).shape == (3,)


def test_baseline_predicts_train_mean(config, data):
    X_tr, X_te, y_tr, _ = DatasetLoader(config).split(*data)
    pipe = Trainer(config).build_pipeline(DummyRegressor()).fit(X_tr, y_tr)
    np.testing.assert_allclose(pipe.predict(X_te), y_tr.mean())


def test_compare_picks_best_and_predictions_in_range(config, data):
    X_tr, X_te, y_tr, y_te = DatasetLoader(config).split(*data)
    trainer = Trainer(config)
    table = trainer.compare(X_tr, y_tr)
    assert len(table) == 3
    assert table["cv_rmse"].is_monotonic_increasing
    preds = trainer.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert ((preds >= 0) & (preds <= 20)).all()
    assert trainer.test_score(X_te, y_te)["rmse"] > 0


def test_save_load_roundtrip(config, data, tmp_path):
    X_tr, X_te, y_tr, _ = DatasetLoader(config).split(*data)
    trainer = Trainer(config)
    trainer.compare(X_tr, y_tr)
    trainer.save(tmp_path / "m.joblib")
    other = Trainer(config)
    other.load(tmp_path / "m.joblib")
    np.testing.assert_allclose(trainer.predict(X_te), other.predict(X_te))
