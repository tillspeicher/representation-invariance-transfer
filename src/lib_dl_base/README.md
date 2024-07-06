# lib_dl_base

A repository for shared core (deep learning) functionality

## Setup

After cloning the repository, run `./setup.sh` to initialize git submodules, install Poetry, the project dependencies, and the pre-commit hooks.

### Dependency Management

The library uses [Poetry](https://python-poetry.org/) to manage dependencies.
Poetry is more explicit than pip about how it keeps track of dependencies and creates virtual environments in away from the actual project, which makes it easier to maintain projects under different environments.
Poetry is installed automatically by running the `setup.sh` script.

### Contributing to the library

The project uses [git pre-commit hooks](https://pre-commit.com/) to maintain code style and integrity, i.e. for code formatting, linting and unit-testing.
Pre-commit hooks are installed automatically by running the `setup.sh` script, and will also run automatically on subsequent commits.

To manually run the pre-commit hooks, execute `poetry run pre-commit run --all-files`.
