import pandas as pd

from student_grades.data import DatasetLoader


def test_clean_drops_missing_target_and_duplicates(config, data):
    X, y = data
    X = pd.concat([X, X.iloc[:5]], ignore_index=True)
    y = pd.concat([y, y.iloc[:5]], ignore_index=True)
    y.iloc[0] = None
    X_c, y_c = DatasetLoader(config).clean(X, y)
    assert len(X_c) == len(y_c)
    assert y_c.notna().all()
    assert not X_c.assign(t=y_c).duplicated().any()


def test_clean_keeps_grades_in_range(config, data):
    X, y = data
    y = y.copy()
    y.iloc[0] = 35
    _, y_c = DatasetLoader(config).clean(X, y)
    assert y_c.between(0, 20).all()


def test_split_has_no_overlap(config, data):
    loader = DatasetLoader(config)
    X_tr, X_te, _, _ = loader.split(*data)
    assert set(X_tr.index).isdisjoint(X_te.index)


def test_split_size_and_reproducibility(config, data):
    loader = DatasetLoader(config)
    X_tr, X_te, _, _ = loader.split(*data)
    assert len(X_te) == round(len(data[0]) * config["test_size"])
    X_tr2, _, _, _ = loader.split(*data)
    assert X_tr.index.equals(X_tr2.index)
