import json

import pytest

from student_grades.cli import main, read_case
from student_grades.data import DatasetLoader
from student_grades.models import Trainer


def make_case(X, path):
    case = json.loads(X.iloc[[0]].to_json(orient="records"))[0]
    path.write_text(json.dumps(case), encoding="utf-8")
    return list(X.columns)


def test_read_case_returns_one_row_in_order(data, tmp_path):
    X, _ = data
    file = tmp_path / "case.json"
    columns = make_case(X, file)
    case = read_case(file, columns[::-1])
    assert case.shape == (1, len(columns))
    assert list(case.columns) == columns[::-1]


def test_read_case_missing_feature_raises(data, tmp_path):
    X, _ = data
    file = tmp_path / "case.json"
    columns = make_case(X, file)
    with pytest.raises(ValueError):
        read_case(file, columns + ["not_a_feature"])


def test_main_requires_a_command():
    with pytest.raises(SystemExit):
        main([])


def test_predict_without_model_returns_error(data, tmp_path, capsys):
    file = tmp_path / "case.json"
    make_case(data[0], file)
    missing_model = tmp_path / "x.joblib"
    code = main(["predict", "--input", str(file), "--model", str(missing_model)])
    assert code == 1
    assert "train" in capsys.readouterr().out


def test_predict_prints_a_grade_between_0_and_20(config, data, tmp_path, capsys):
    X_tr, _, y_tr, _ = DatasetLoader(config).split(*data)
    trainer = Trainer(config)
    trainer.compare(X_tr, y_tr)
    model_file = tmp_path / "m.joblib"
    trainer.save(model_file)
    case_file = tmp_path / "case.json"
    make_case(data[0], case_file)
    code = main(["predict", "--input", str(case_file), "--model", str(model_file)])
    out = capsys.readouterr().out
    assert code == 0
    grade = float(out.split(":")[1].split("/")[0])
    assert 0 <= grade <= 20