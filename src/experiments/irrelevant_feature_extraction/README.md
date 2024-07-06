## (Ir-)Relevant Feature Invariance

Do models capture/retain information about features in their representations if the features are present in the data, but not relevant for solving the objective that they are trained for?

We train a model on CIFAR data with one of n objects pasted into the images randomly. Then we evaluate transfer performance on the task of predicting the random objects, and vice versa.

## Usage

To run the experiment, use the following command:

```bash
poetry run python src/main.py +ife=<config_name>
```

where `config_name` can be e.g. `test` or `rn18_fixed`.
See `configs.py` for all available configurations.
