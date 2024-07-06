import argparse
import logging
import os
import subprocess
import sys
from typing import Any, cast

from hydra.core.config_store import ConfigStore

from lib_dl_base.io.dirs import set_artifacts_dir, set_dataset_dir

from .config_defs import ProjectConfig, ServerConfig, UploadConfig
from .experiment import C, ExperimentHandle
from .notebook.upload import register_upload_config, set_upload_config


logger = logging.getLogger(__name__)


class ExperimentConfigAction(argparse.Action):
    def __call__(self, parser, namespace, values: str, option_string=None):
        if values == "nb":
            setattr(namespace, self.dest, None)
            return
        if not (values.startswith("+") and "=" in values):
            parser.error(
                "Main argument must be in the format +<exp_id>=<config_name>"
            )
        name, value = values.split("=", 1)
        setattr(namespace, self.dest, (name[1:], value))


def setup_project(
    experiment_handles: list[ExperimentHandle[C]],
) -> None:
    cs = ConfigStore.instance()
    register_upload_config()

    experiment_id_map = {handle.id: handle for handle in experiment_handles}

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exp_id",
        type=str,
        help="Experiment ID",
        # choices=experiment_id_map.keys(),
        action=ExperimentConfigAction,
    )
    args, remaining_args = parser.parse_known_args()
    arg_values = getattr(args, "exp_id", None)
    if arg_values is not None:
        exp_id, config_name = arg_values
        if exp_id not in experiment_id_map:
            parser.error(
                f"Invalid experiment ID: {exp_id}\n"
                f"Valid IDs: {list(experiment_id_map.keys())}\n"
                "Or use 'nb' to start a Jupyter notebook server"
            )

        target_exp_handle = experiment_id_map[exp_id]
        for config in target_exp_handle.create_configs():
            if config.name == config_name:
                cs.store(
                    group=target_exp_handle.id,
                    name=config.name,
                    node=config,
                )
                break
        else:
            parser.error(
                f"Invalid config name: {config_name}\n"
                f"Valid names: {', '.join(c.name for c in target_exp_handle.create_configs())}"
            )
        exp_arg = [f"+{exp_id}={config_name}"]
    else:
        exp_arg = []

    # TODO: passing the experiment id - config name this way is kinda hacky
    sys.argv = sys.argv[:1] + exp_arg + remaining_args


def run_project(
    config: ProjectConfig,
    experiment_handles: list[ExperimentHandle[C]],
    handle_args: dict[str, Any] = {},
) -> None:
    set_artifacts_dir(config.dirs.artifacts)
    set_dataset_dir(config.dirs.data)
    set_upload_config(config.server, config.upload)

    for handle in experiment_handles:
        experiment_config = getattr(config, handle.id, None)
        if experiment_config is not None:
            experiment_config = handle.set_seed(experiment_config, config.sid)
            for arg_name, arg in handle_args.items():
                setattr(experiment_config, arg_name, arg)
            iter = handle.experiment(experiment_config)
            try:
                # If the experiment returns an iterable, iterate over it
                for _ in iter:
                    pass
            except TypeError:
                pass
            return

    logger.info("No experiment config provided. Starting notebook server...")
    nb_port = config.ports.jupyter
    upload_config = cast(dict, config.upload)
    server_config = cast(dict, config.server)
    subprocess.run(
        [
            "python",
            "-m",
            # "poetry",
            # "run",
            "jupyter",
            "lab",
            "--allow-root",
            "--no-browser",
            "--ip=0.0.0.0",
            f"--port={nb_port}",
        ],
        env={
            # Pass the current path to the subprocess, esp. the
            # project-specific libraries that might have been added
            **os.environ,
            "PYTHONPATH": ":".join(sys.path),
            **UploadConfig(**upload_config).to_env_vars(),
            **ServerConfig(**server_config).to_env_vars(),
        },
    )


# local_rank = -1


# def parse_local_rank() -> int:
#     """Parse the --local_rank=<num> argument set by deepseed.
#     Forward the rest of the arguemnts to Hydra."""
#     global local_rank
#     remaining_args = []
#     for arg in sys.argv:
#         if arg.startswith("--local_rank="):
#             local_rank = int(arg[len("--local_rank=") :])
#             os.environ["LOCAL_RANK"] = str(local_rank)
#         else:
#             remaining_args.append(arg)
#     sys.argv = remaining_args
#     return local_rank
