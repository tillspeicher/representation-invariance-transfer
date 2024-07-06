import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Union

import yaml

from ..config_defs import ProjectConfig
from .connect_server import add_connect_parser
from .docker.build import add_build_docker_parser
from .docker.run import add_run_docker_parser
from .init_config import INIT_COMMAND, ProjectDefaults, add_config_init_parser
from .slurm.run_slurm import add_slurm_parser
from .sync import add_sync_parser


CONFIG_FILE_NAME = "project_config.yaml"


def run_actions(
    defaults: ProjectDefaults,
    config_file: Union[str, Path] = CONFIG_FILE_NAME,
) -> None:
    parser = argparse.ArgumentParser(description="Run project actions")
    subparsers = parser.add_subparsers(dest="command")
    add_sync_parser(subparsers)
    add_connect_parser(subparsers)
    add_build_docker_parser(subparsers)
    add_run_docker_parser(subparsers)
    add_config_init_parser(subparsers, defaults)
    add_slurm_parser(subparsers)
    args, unknown_args = parser.parse_known_args()

    if args.command == INIT_COMMAND:
        if Path(config_file).exists():
            raise FileExistsError(
                f"Config file {config_file} already exists. "
                "Delete it to create a new one."
            )
        config = args.func(None, args, unknown_args)
        with open(config_file, "w") as f:
            yaml.dump(asdict(config), f, sort_keys=False)
        print(f"Config file {config_file} created.")
        if defaults.notice:
            print(
                "NOTICE:",
                defaults.notice.format(username=config.server.username),
            )
    else:
        try:
            with open(config_file, "r") as f:
                config_dict = yaml.safe_load(f)
                project_config = ProjectConfig.from_dict(config_dict)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file {config_file} not found. "
                f"Run the script with the '{INIT_COMMAND}' command "
                "first to create it."
            )
        args.func(project_config, args, unknown_args)
