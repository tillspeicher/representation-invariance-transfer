# Code for "Understanding the Role of Invariance in Transfer Learning", [TMLR 2024]()

This code was used for running the experiments in the paper [Understanding the Role of Invariance in Transfer Learning]().

The paper explores how important representational invariance for transfer performance, compared to other factors such as model architecture and pretraining dataset size, under which conditions invariance can harm transfer performance, and how transferable invariance is to other domains.
Experiments are conducted using a synthetic dataset, Transforms-2D, with carefully controlled transformations acting on image objects, as well as with augmented CIFAR datasets.

## Usage

The code uses [Poetry](https://python-poetry.org/) to manage dependencies.
Dependencies can be installed from the `pyproject.toml` file by running `poetry install`.

Before running the code, generate a configuration file by running `python actions.py init`.
Some of the configurations are specific to the original environment, so you may have to tweak the `actions.py` script a bit.

Experiments can be run by executing commands of the form `poetry run python src/main.py +<experiment_id>=<config_name> [++<option>=<value> ...]`.
The code for the different experiments can be found inside `src/experiments`.
Each experiment has a README file that contains additional information and instructions for how to run it.
There are four main experiments:

- [Transforms/Invariance vs other factors](./src/experiments/transforms_vs_other/)
- [Invariance to (ir-)relevant features](./src/experiments/irrelevant_feature_extraction/)
- [Invariance transfer](./src/experiments/invariance_transfer/)
- [Invariance mismatch](./src/experiments/transforms_mismatch/)

To start a Jupyter notebook server, run `poetry run python src/main.py nb`.

The output of experiments (checkpoints, logs, results) are stored inside the `artifacts` directory.

## Transforms-2D Dataset

The experiments use the [Transforms-2D](./src/transforms_2d/) dataset, located under `src/transforms_2d`.
The dataset consists of transformed versions of image objects with transparency masks, pasted onto background images.
See the datasets's [README](./src/transforms_2d/README.md) for more information.

The base images used by Transforms-2D are available on the [Huggingface Hub](https://huggingface.co/datasets/tillspeicher/transforms_2d_base).
They are based on the [SI-score dataset](https://github.com/google-research/si-score/).
Please consider citing the original authors of this work as well, if you use these images.

## Citation
