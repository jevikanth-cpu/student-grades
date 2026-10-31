"""Command-line interface: train, evaluate and predict."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from student_grades.config import PROJECT_ROOT, load_config
from student_grades.data import DatasetLoader
from student_grades.models import Trainer


def default_model_path(config: dict[str, Any]) -> Path:
    """Return the model path from the config, relative to the project root."""
    return PROJECT_ROOT / config["model_path"]


def read_case(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read one new case from a JSON file.

    Args:
        path: JSON file with one object (feature name -> value).
        columns: Feature names the model expects.

    Returns:
        A one-row DataFrame with the columns in the expected order.

    Raises:
        ValueError: If some expected features are missing from the file.
    """
    with open(path, encoding="utf-8") as handle:
        case = json.load(handle)
    missing = [col for col in columns if col not in case]
    if missing:
        raise ValueError(f"Missing features in {path.name}: {missing}")
    return pd.DataFrame([case])[columns]


def cmd_train(args: argparse.Namespace) -> int:
    """Train, compare the models, and save the best one."""
    config = load_config()
    X_train, _, y_train, _ = DatasetLoader(config).get_splits()
    trainer = Trainer(config)
    print(trainer.compare(X_train, y_train).to_string())
    path = args.model or default_model_path(config)
    trainer.save(path)
    print(f"Saved the {trainer.best_name} model to {path}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Print the test metrics of the saved model."""
    config = load_config()
    path = args.model or default_model_path(config)
    if not path.exists():
        print(f"No model at {path}. Run the train command first.")
        return 1
    _, X_test, _, y_test = DatasetLoader(config).get_splits()
    trainer = Trainer(config)
    trainer.load(path)
    metrics = trainer.test_score(X_test, y_test)
    print(f"Test RMSE: {metrics['rmse']:.2f}   Test MAE: {metrics['mae']:.2f}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Predict the final grade of one new student from a JSON file."""
    config = load_config()
    path = args.model or default_model_path(config)
    if not path.exists():
        print(f"No model at {path}. Run the train command first.")
        return 1
    trainer = Trainer(config)
    trainer.load(path)
    columns = list(trainer.best_model.named_steps["prep"].feature_names_in_)
    try:
        case = read_case(args.input, columns)
    except ValueError as error:
        print(error)
        return 1
    grade = float(trainer.predict(case)[0])
    print(f"Predicted final grade (G3): {grade:.1f} / 20")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with the three commands."""
    parser = argparse.ArgumentParser(
        prog="student-grades", description="Predict the final grade of a student."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func, help_text in [
        ("train", cmd_train, "train the models and save the best one"),
        ("evaluate", cmd_evaluate, "print the test metrics of the saved model"),
        ("predict", cmd_predict, "predict the grade of one new student"),
    ]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--model", type=Path, help="model file (default: config)")
        command.set_defaults(func=func)
        if name == "predict":
            command.add_argument(
                "--input", type=Path, required=True, help="JSON file with one case"
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of the command-line tool."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())