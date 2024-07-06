import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Union

from ..config_defs import (
    DirectoryConfig,
    DockerBuildConfig,
    DockerConfig,
    PortsConfig,
    ProjectConfig,
    ServerConfig,
    SyncConfig,
    UploadConfig,
)


INIT_COMMAND = "init"
DEFAULT_RESULTS_ROOT_DIR = "/home/{username}/results/{project_name}"
DEFAULT_RESULTS_URL_PREFIX = "<server_url>/{username}/results/{project_name}"


@dataclass
class ProjectDefaults:
    project_name: str
    dockerfile: str
    docker_image_storage_dir: str
    docker_volumes: list[str]
    docker_environment: list[str]
    docker_env_file: list[Union[str, Path]]
    server_project_root: str = "<server_path>/{username}/{project_name}"
    notice: Optional[str] = None


def add_config_init_parser(
    subparsers: argparse._SubParsersAction,
    defaults: ProjectDefaults,
) -> None:
    parser = subparsers.add_parser(
        INIT_COMMAND,
        help="Initialize a new config file",
        description="Initialize a new config file",
    )
    parser.set_defaults(func=partial(init_config, defaults))


def init_config(
    defaults: ProjectDefaults,
    config: Optional[ProjectConfig],
    args: argparse.Namespace,
    unkonwn_args: list[str],
) -> ProjectConfig:
    # Create the project config in an interactive way with defaults
    username = input("Enter your username: ")

    default_server_project_root = defaults.server_project_root.format(
        username=username,
        project_name=defaults.project_name,
    )
    results_root_dir = (
        input(
            "Enter the path to the results directory on the server: "
            f"(default: {default_server_project_root}) "
        )
        or default_server_project_root
    )

    default_results_url_prefix = DEFAULT_RESULTS_URL_PREFIX.format(
        username=username,
        project_name=defaults.project_name,
    )
    results_url_prefix = (
        input(
            "Enter the URL prefix for the results directory: "
            f"(default: {default_results_url_prefix}) "
        )
        or default_results_url_prefix
    )

    server_project_root = (
        input(
            "Enter the path to the project root on the server: "
            f"(default: {default_server_project_root}) "
        )
        or default_server_project_root
    )

    image_storage_dir = defaults.docker_image_storage_dir.format(
        username=username,
    )

    return ProjectConfig(
        dirs=DirectoryConfig(),
        ports=PortsConfig(),
        server=ServerConfig(
            username=username,
        ),
        upload=UploadConfig(
            results_root_dir=results_root_dir,
            results_url_prefix=results_url_prefix,
        ),
        sync=SyncConfig(server_project_root=server_project_root),
        docker=DockerConfig(
            image=defaults.project_name,
            build=DockerBuildConfig(
                dockerfile=defaults.dockerfile,
            ),
            image_storage_dir=image_storage_dir,
            volumes=defaults.docker_volumes,
            environment=defaults.docker_environment,
            env_file=defaults.docker_env_file,
        ),
    )
