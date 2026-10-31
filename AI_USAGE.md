# AI usage

I used Claude (Anthropic's chat assistant) as a helper during this project, working alone. This file lists what I used it for. I ran and checked every piece of code myself.

| Part | What Claude did | What I did / changed |
|---|---|---|
| Package skeleton (`data.py`, `features.py`, `models.py`, `evaluate.py`, `config.py`) | Proposed the structure and wrote a first version of the `DatasetLoader` and `Trainer` classes and the pipeline | Created the files, ran the tests and the real data download on my machine, and fixed problems |
| Tests (`tests/`) | Wrote a first version of the 14 tests | Ran them, fixed the `conftest.py` and `test_models.py` updates for Huber, and fixed a ruff warning |
| Command-line tool (`cli.py`) | Wrote the `train`, `evaluate` and `predict` commands and the example JSON case | Created the files, fixed a misplaced `examples` folder, ran all three commands, and tested a changed case (3 failures gives a lower grade) |
| Notebook (`analysis.ipynb`) | Suggested the analysis steps and the code of each cell, and helped me interpret the outputs | Ran every cell, widened the Ridge grid after seeing alpha at the edge, decided to keep the G3 = 0 students, and decided to exclude G1 and G2 |
| Huber model | Suggested adding Huber loss after I saw that the G3 = 0 students cause about 31% of the squared error | Added it to the config and `Trainer`, and compared it under the same CV |
| Notebook text and README | Drafted the explanation text and the README | Checked the numbers against my own outputs |
| Git and VS Code | Explained the commands step by step | Ran all the Git commands myself |

## What I can explain

I can explain every line of the code I submit and every number in the notebook. Anything I cannot explain at the defense should be counted as not done.