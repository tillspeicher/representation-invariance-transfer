# Invariance Transfer Across Domains

We measure how invariant the representations of a model trained for one transformation are to another transformation.
We do so for each combination of transformations, i.e. we evaluate the invariance models trained on each transformation on each other transformation.
This helps us understand what inductive biases models have towards acquiring different invariances.

## Methodology

For each transformation, we create a dataset with only that transformation acting on the objects.
We use these datasets to train a model for each transformation and then evaluate its invariance (and performance) for each other transformation.

We measure representational invariance by computing the similarity resp. distance between penultimate layer representations of inputs that only differ in the transformation that is applied to them. E.g. for translation, we would measure the similarity between two representations of the same object on the same background, but translated to different positions within the image.

We use l2-distance to measure invariance.
We normalize the l2-distance values by dividing them by the average l2-distance of representations for random images in the test set.
I.e. the distance values should be interpreted as "how different are representation of input pairs that only differ in their transformation, relative to random pairs of images".

## Usage

To run the experiment, use the following command:

```bash
poetry run python src/main.py +it=<config_name>
```

where `config_name` can be e.g. `test` or `rn18`.
See `configs.py` for all available configurations.
