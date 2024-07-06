# lib_project

A library for common code needed to manage projects.

## Setup

IMPORTANT: Clone the project via SSH (rather than HTTPS), since otherwise there will be issues pulling the submodules.
You need to register an SSH key ([see Gitlab instructions](https://docs.gitlab.com/ee/user/ssh.html)), in order for this to work.

After cloning the repository, run `./setup.sh` to initialize git submodules, install Poetry, the project dependencies, and the pre-commit hooks.

You can pull the latest updates (including from the submodule dependencies) using `git pull --recurse`.

### Dependency Management

The library uses [Poetry](https://python-poetry.org/) to manage dependencies.
Poetry is more explicit than pip about how it keeps track of dependencies and creates virtual environments in away from the actual project, which makes it easier to maintain projects under different environments.
Poetry is installed automatically by running the `setup.sh` script.

### Contributing to the library

The project uses [git pre-commit hooks](https://pre-commit.com/) to maintain code style and integrity, i.e. for code formatting, linting and unit-testing.
Pre-commit hooks are installed automatically by running the `setup.sh` script, and will also run automatically on subsequent commits.

To manually run the pre-commit hooks, execute `poetry run pre-commit run --all-files`.
