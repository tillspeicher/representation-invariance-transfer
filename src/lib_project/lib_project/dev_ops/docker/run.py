import argparse
import subprocess
from pathlib import Path
from typing import Union

from ...config_defs import DockerConfig, PortsConfig, ProjectConfig
from ..utils import run_command
from .docker_utils import (
    CONTAINER_HOME_DIR,
    get_gpu_ids,
    get_image_hash,
    get_image_name,
    get_logger,
    get_nfs_image_hash_path,
    get_nfs_image_path,
    is_running_on_slurm,
    prefix_list_elements,
)


"""
This script runs
the commands provided to it, or starts an interactive shell session
if there are none.
You can specify the GPUs it should use via the --gpus id1,id2,... flag.
If no GPUs are specified, it will only use the CPU.
"""

logger = get_logger(__name__)


CWD = Path.cwd()
DEFAULT_DIRECTORIES_TO_MOUNT = [
    f"{CWD / dir}:{CONTAINER_HOME_DIR / dir}"
    for dir in [
        "src",
        "conf",
        "artifacts",
        "data",
        "logs",
        "project_config.yaml",
    ]
]


def add_run_docker_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "run-docker",
        help="Run the project as a Docker container",
    )
    parser.add_argument(
        "--devel",
        action="store_true",
        default=False,
        help="Use the (larger) devel container instead of the runtime one.",
    )
    parser.add_argument(
        "--gpus",
        default="",
        type=str,
        help="List of GPU IDs to use.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        default=False,
        help="Start the container in interactive mode, i.e. for shell sessions and Jupyter.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variables to pass to the container.",
    )
    parser.add_argument(
        "--no-progressbar",
        action="store_true",
        default=False,
        help="Don't show tqdm progress bars.",
    )
    parser.add_argument(
        "--jupyter-port",
        type=int,
        default=None,
        help="The port to use for Jupyter.",
    )
    parser.set_defaults(func=run_container)


def run_container(
    config: ProjectConfig,
    args: argparse.Namespace,
    remaining_args: list[str],
) -> None:
    username = config.server.username
    image_name = get_image_name(config.docker.image, args.devel)

    load_command = _get_load_command(
        username,
        config.docker.image_storage_dir,
        image_name,
    )
    if not run_command(load_command):
        logger.error("Loading the image failed.")
        return

    interactive = args.interactive
    if len(remaining_args) > 0:
        container_command = " ".join(remaining_args)
    else:
        container_command = "bash"
        interactive = True
    exec_command = _get_exec_command(
        config.docker,
        config.ports,
        args,
        image_name=image_name,
        interactive=interactive,
    )
    # Start the SSH agent if we are running in interactive mode
    # It's needed to publish results by uploading HTML pages via SSH
    if interactive:
        ssh_start_command = f"{_get_ssh_agent_start_command()};"
    else:
        ssh_start_command = ""

    full_command = f"{ssh_start_command} {exec_command} {container_command}"
    # print(full_command)
    if not run_command(full_command):
        logger.error("Executing the container failed.")
        return


def _get_load_command(
    username: str,
    image_storage_dir: Union[str, Path],
    image_name: str,
) -> str:
    if is_running_on_slurm():
        # Speed up container loading on SLURM machines by using
        # a memcached docker directory. All Docker data will be removed
        # again on these machines after jobs have finished.
        shared_mem_docker_dir = Path("f/dev/shm/docker_data_{username}")
        shared_mem_docker_dir.mkdir(parents=True, exist_ok=True)
        docker_data_dir = Path(f"/tmp/docker_data_{username}")
        docker_data_dir.symlink_to(
            shared_mem_docker_dir, target_is_directory=True
        )
    # Read the content of the hash file
    image_hash_path = get_nfs_image_hash_path(image_storage_dir, image_name)
    if not image_hash_path.exists():
        logger.warning(
            f"Image hash file {image_hash_path} does not exist. "
            "Skipping image loading."
        )
        return ""

    remote_hash = image_hash_path.read_text().strip()
    if not _image_exists_locally(image_name) or remote_hash != get_image_hash(
        image_name
    ):  # The image has changed
        image_storage_path = get_nfs_image_path(image_storage_dir, image_name)
        logger.info(
            f"Image '{image_name}' not found locally or it has changed. "
            f"Loading it from {image_storage_path}..."
        )
        return f"time docker load < {image_storage_path}"
    else:
        return ""


def _image_exists_locally(image_name: str) -> bool:
    cmd = f"docker images -q {image_name}"
    try:
        output = subprocess.check_output(cmd, shell=True, text=True)
        return len(output.strip()) > 0
    except subprocess.CalledProcessError:
        return False


def _get_ssh_agent_start_command() -> str:
    """Construct a command to start the SSH agent if it's not running."""
    return (
        "[ -z $(env | grep 'SSH_AUTH_SOCK') ] "
        "&& echo 'WARNING: SSH agent not running, starting without SSH "
        "functionality. If you with to start SSH connections from within "
        "the container, e.g. for uploading results, start it manually via "
        "`eval $(ssh-agent -s)` and restart the container.'"
    )
    # return (
    #     # "[ \"$(env | grep 'SSH_AUTH_SOCK')\" ] || [ -z '$SSH_AUTH_SOCK' ] "
    #     "[ -z $(env | grep 'SSH_AUTH_SOCK') ] "
    #     "&& echo 'SSH agent not running, starting.' && eval $(ssh-agent -s)"
    # )


def _get_exec_command(
    docker_config: DockerConfig,
    ports_config: PortsConfig,
    args: argparse.Namespace,
    image_name: str,
    interactive: bool,
) -> str:
    gpu_ids = get_gpu_ids(args.gpus)
    volumes = prefix_list_elements(
        "-v",
        DEFAULT_DIRECTORIES_TO_MOUNT + docker_config.volumes,
    )
    container_type = "devel" if args.devel else "runtime"
    environment_variables = prefix_list_elements(
        "-e",
        (
            _parse_env_files(docker_config.env_file)
            + docker_config.environment
            + args.env
        ),
    )

    if interactive:
        if args.jupyter_port is not None:
            ports_config.jupyter = args.jupyter_port
        port_args = [f"-p={port}:{port}" for port in ports_config.as_list()]
        ssh_agent_args = [
            "-v",
            "${SSH_AUTH_SOCK}:/ssh-agent",
            "-e",
            "SSH_AUTH_SOCK=/ssh-agent",
            "-v",
            f"$HOME/.ssh/known_hosts:{CONTAINER_HOME_DIR / '.ssh/known_hosts'}",
        ]
        # Set Jupyter directories to persist sessions
        # See https://docs.jupyter.org/en/latest/use/jupyter-directories.html
        jupyter_args = [
            "-v",
            f"$HOME/.jupyter:{CONTAINER_HOME_DIR / '.jupyter'}",
            "-v",
            (
                "$HOME/.local/share/jupyter:"
                + str(CONTAINER_HOME_DIR / ".local/share/jupyter")
            ),
            "-e",
            "JUPYTER_PATH=" + str(CONTAINER_HOME_DIR / ".jupyter"),
        ]
        # interaction_args = (
        #     ["-it", "--ip=0.0.0.0"] + port_args + ssh_agent_args + jupyter_args
        # )
        interaction_args = (
            [
                "-it",
                # "--ip=0.0.0.0",
            ]
            + port_args
            + ssh_agent_args
            + jupyter_args
        )
    else:
        # Disable exposed ports and interactivity if not running from a TTY
        interaction_args = []

    # The Docker setup here is run both with rootful and rootless Docker
    # and we only want to switch to a non-root user in the rootful case
    # as the rootless one does it automatically, and doing an additional
    # switch causes permission errors when creating or accessing files.
    # See https://sthbrx.github.io/blog/2023/04/05/detecting-rootless-docker/
    user_args = (
        []
        if _deamon_is_rootless()
        else ["-u", "$(id -u):$(id -g)"]  # Run as current user
    )
    container_type_args = ["-e", f"CONTAINER_TYPE={container_type}"]
    gpu_args = ["--gpus", f"'\"device={gpu_ids}\"'"] if len(gpu_ids) > 0 else []

    exec_command = " ".join(
        [
            "docker",
            "run",
            "--rm",
            *interaction_args,
            *container_type_args,
            *volumes,
            "--shm-size",
            "128gb",
            *gpu_args,
            # "-e",
            # "CUDA_VISIBLE_DEVICES=" + gpu_ids,
            *user_args,
            *(["-e", "DISABLE_PROGRESSBAR=1"] if args.no_progressbar else []),
            *environment_variables,
            image_name,
        ]
    )
    return exec_command


def _deamon_is_rootless() -> bool:
    cmd = "docker info -f '{{println .SecurityOptions}}'"
    try:
        output = subprocess.check_output(cmd, shell=True, text=True)
        return "name=rootless" in output
    except subprocess.CalledProcessError:
        return False


def _parse_env_files(env_files: list[Union[str, Path]]) -> list[str]:
    env_vars = []
    for env_file in env_files:
        env_file = str(env_file).format(home=Path.home())
        with open(env_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) > 0 and not line.startswith("#"):
                    env_vars.append(line)
    return env_vars
