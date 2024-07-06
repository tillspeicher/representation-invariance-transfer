# lib_vision

A library with common DL vision model architectures and training schemes, built around [Pytorch Lightning](https://www.pytorchlightning.ai/).
The goal of this library is to provide templates and boilerplate code for creating DL models, to get started more quickly with experiments.

A library with utilities to run analysis tasks in DL vision projects.
The goal of the library is to provide a repository of commonly used analysis functionality such as visualization, etc.

## Usage

### Adding the library to a project

Run `git submodule add <library_ssh_url> <local_path>` to add the submodule into a project.

Run `git submodule update --remote` in the parent project to fetch the latest changes.

### Setup

The project uses [Poetry](https://python-poetry.org/) to manage dependencies.
Poetry is more explicit than pip about how it keeps track of dependencies to facilitate deterministic builds and creates virtual environments away from the actual project, which makes it easier to maintain projects under different environments.
To install Poetry, consult [this guide](https://python-poetry.org/docs/master/#installing-with-the-official-installer).

To install the library dependencies, run `poetry install`.
Then, run `poetry run pre-commit install` to setup the pre-commit hooks for code formatting, linting and unit-testing.

To manually run the pre-commit hooks, execute `poetry run pre-commit run --all-files`.

## Design

### Models

This project wraps Pytorch models with the code in `wrappers`, such that models produce `InferenceRecord`s instead of the normal plain logit tensor.
This is done so that the output of models can be easily extended without having to adapt existing code, for instance when the activations of intermediate layers are needed along with the final model output.

### Datasets

This project wraps Pytorch Datasets with the code in `wrappers`, such that they produce `DataSample`s instead of the normal tuple of `x` and `y` tensors.
This is done to make it to pass additional data along with the dataset samples without breaking existing code, such as hierarchical information, metadata, etc.
