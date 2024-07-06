#!/usr/bin/env python3

import sys
from pathlib import Path


ROOT_PATH = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_PATH / "src/lib_project/"))

from lib_project.dev_ops.actions import ProjectDefaults, run_actions

# TODO: the configuration here is specific to the original environment
# the project was run in, you will probably have to tweak this part and
# teh code in lib_project it uses to make it work.
PROJECT_DEFAULTS = ProjectDefaults(
    project_name="representation_invariance",
    dockerfile="scripts/Dockerfile",
    docker_image_storage_dir="",
    docker_volumes=[],
    docker_environment=[],
    docker_env_file=[],
    server_project_root="<server>/{username}/{project_name}",
    notice="",
)


if __name__ == "__main__":
    run_actions(defaults=PROJECT_DEFAULTS)
