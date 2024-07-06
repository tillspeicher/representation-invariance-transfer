# Transforms/Invariance Mismatch Experiment

This experiment aims to understand how representations transfer to downstream tasks when there is a mismatch between the invariances learned during pretraining and those needed on the target task.

We train models on nested combinations of transformations, ranging from 1 to 8 transformations.
I.e. the 3 transformation combination uses the same transformations as the 2 transformation combination, plus one additional transformation type.
Then, we test how well models trained on one set of transformations transfer to another dataset with a sub- or superset of transformations.

## Usage

To run the experiment, use the following command:

```bash
poetry run python src/main.py +tm=<config_name>
```

where `config_name` can be e.g. `test` or `rn18`.
See `configs.py` for all available configurations.
