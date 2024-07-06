#!/bin/bash

# Setup submodules
git submodule update --init --recursive

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
# Install dependencies
poetry install --with dev

# Install pre-commit hooks
poetry run pre-commit install

# Link the virtual environment, for pyright
ln -si $(poetry env info --path) .venv
