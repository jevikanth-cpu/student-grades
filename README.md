# student-grades

A small command-line tool that predicts the final grade (G3, from 0 to 20) of a secondary-school student, using the UCI *Student Performance* dataset. It is the final project of the Machine Learning and Advanced Python courses (PGE 2).

The tool is meant for an early warning: it does **not** use the earlier period grades G1 and G2, so it can run before any grade exists. On the held-out test set, the chosen model (Huber regression) has an RMSE of about 2.85 and an MAE of about 2.03 grade points.

## Install

Python 3.10 or newer is required.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac / Linux
pip install -e ".[dev]"
```

## Run the tests

```bash
pytest
```

## Train, evaluate, predict

```bash
student-grades train
student-grades evaluate
student-grades predict --input examples/new_case.json
```

- `train` downloads the dataset, compares four models with cross-validation, and saves the best one to `models/model.joblib`.
- `evaluate` prints the test RMSE and MAE of the saved model (the test set is used only here).
- `predict --input FILE.json` loads the saved model and prints the predicted final grade for one student. The JSON file must contain the 30 features (see `examples/new_case.json`).

If the `student-grades` command is not found, use `python -m student_grades.cli` instead (for example `python -m student_grades.cli train`).

## Project structure

```
config.yaml              seed, split size, CV settings, hyperparameter grids
src/student_grades/
    data.py              DatasetLoader: download, clean, split
    features.py          ColumnTransformer (imputation, scaling, encoding)
    models.py            Trainer: pipelines, CV, comparison, save, load
    evaluate.py          metrics (RMSE, MAE)
    cli.py               train, evaluate, predict
tests/                   pytest tests
notebooks/analysis.ipynb the full modelling story
examples/new_case.json   one example student for the predict command
models/                  saved model (joblib)
```

## Data and credits

- Dataset: P. Cortez, *Student Performance*, UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/320, licence CC BY 4.0.
- Related paper: P. Cortez and A. Silva, *Using Data Mining to Predict Secondary School Student Performance*, FUBUTEC 2008.
- Libraries: scikit-learn, pandas, numpy, joblib, PyYAML, ucimlrepo.
- Use of generative AI tools is declared in `AI_USAGE.md`.