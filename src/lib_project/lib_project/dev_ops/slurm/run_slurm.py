import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ...config_defs import ProjectConfig


@dataclass
class SlurmConfig:
    partition: str
    gpus: int
    cpus: int
    memory: str
    nodes: int = 1
    cpu_offload: bool = False

    def to_cmd_args(self) -> list[str]:
        cmd = [
            f"--partition={self.partition}",
            f"--nodes={self.nodes}",
            f"--gres=gpu:{self.gpus}",
            f"--cpus-per-task={self.cpus}",
            f"--mem={self.memory}",
        ]
        return cmd


FILE_DIR = Path(__file__).parent


NODE_CONFIGS = {
    "1a40": SlurmConfig(
        partition="a40",
        gpus=1,
        cpus=8,
        memory="96G",
    ),
    "1a40-ol": SlurmConfig(
        # Should work with Llama2-7B on memorizing 1 1024 token string
        partition="a40",
        gpus=1,
        cpus=14,
        memory="148G",
        cpu_offload=True,
    ),
    "2a40": SlurmConfig(
        partition="a40",
        gpus=2,
        cpus=16,
        memory="96G",
    ),
    "2a40-ol": SlurmConfig(
        partition="a40",
        gpus=2,
        cpus=16,
        memory="244G",
        cpu_offload=True,
    ),
    "3a40": SlurmConfig(
        partition="a40",
        gpus=3,
        cpus=18,
        memory="96G",
    ),
    "1a100": SlurmConfig(
        partition="a100",
        gpus=1,
        cpus=12,
        memory="148G",
    ),
    "1a100-ol": SlurmConfig(
        # 244GB Work with Llama2-13B on memorizing 1 1024 token string, with
        # prefix mappings, and are also the max supported value. 240GB
        # is insufficient.
        partition="a100",
        gpus=1,
        cpus=16,
        memory="244G",
        cpu_offload=True,
    ),
    "2a100": SlurmConfig(
        partition="a100",
        gpus=2,
        cpus=16,
        # memory="196",
        memory="244G",
    ),
    "2a100-ol": SlurmConfig(
        partition="a100",
        gpus=2,
        cpus=16,
        memory="480G",
        cpu_offload=True,
    ),
    "4a100": SlurmConfig(
        partition="a100",
        gpus=4,
        cpus=32,
        # memory="196",
        memory="244G",
    ),
    "4a100-ol": SlurmConfig(
        partition="a100",
        gpus=4,
        cpus=32,
        memory="960G",
        cpu_offload=True,
    ),
    "8a100": SlurmConfig(
        partition="a100",
        gpus=8,
        cpus=32,
        memory="244G",
    ),
    "8a100-ol": SlurmConfig(
        partition="a100",
        gpus=8,
        cpus=48,
        memory="1920G",
        cpu_offload=True,
    ),
}
DURATIONS = {
    "1h": "0-1:00",
    "12h": "0-12:00",
    "1d": "1-0:00",
    "4d": "4-0:00",
}


def add_slurm_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "run-slurm",
        help="Run a job on the SLURM cluster.",
    )
    parser.add_argument(
        "-c",
        "--config",
        choices=list(NODE_CONFIGS.keys()),
        help="The node config to request.",
    )
    parser.add_argument(
        "-s",
        "--sids",
        nargs="+",
        default=None,
        help="The sids to run.",
    )
    parser.add_argument(
        "-t",
        "--time",
        choices=list(DURATIONS.keys()),
        default="12h",
        help="The runtime to request.",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Run the job without docker.",
    )
    parser.add_argument(
        "--wrapper",
        choices=["docker", "docker-devel", "none"],
        default="none",
        help="The wrapper to use for the job.",
    )
    parser.set_defaults(func=run_on_slurm)


def run_on_slurm(
    config: ProjectConfig,
    args: argparse.Namespace,
    remaining_args: list[str],
) -> None:
    if args.config is None:
        raise ValueError(
            "Please specify a node config using the `-c` flag. "
            f"Available configs: {list(NODE_CONFIGS.keys())}"
        )

    node_config = NODE_CONFIGS[args.config]
    runtime = DURATIONS[args.time]

    job_name = _get_job_name(remaining_args)
    log_name = f"logs/{job_name}"
    if args.sids is not None and len(args.sids) > 0:
        log_name += "_sid=%a.out"
    else:
        log_name += ".out"
    command = (
        f"bash {FILE_DIR}/entrypoint.sh "
        + f"{args.wrapper} "
        + f"{1 if node_config.cpu_offload else 0} "
        + " ".join(remaining_args)
    )

    sbatch_prefix = [
        "sbatch",
        *node_config.to_cmd_args(),
        f"--time={runtime}",
        f"--output={log_name}",
        f"--error={log_name}",
        f"--job-name={job_name}",
    ]
    if args.sids is not None and len(args.sids) > 0:
        sbatch_prefix.append(f"--array={','.join(args.sids)}")
        command += " ++sid=$SLURM_ARRAY_TASK_ID"
    sbatch_command = [
        *sbatch_prefix,
        "--wrap",
        command,
    ]
    # print(" ".join(sbatch_command))
    subprocess.call(sbatch_command)


def _get_job_name(job_args: list[str]) -> str:
    name_parts = []
    sid_part = None
    for arg in job_args:
        if arg.startswith("++sid="):
            sid_part = arg[2:]
        elif arg.startswith("+"):
            name_parts.append(arg[1:])
    if len(name_parts) == 0:
        raise ValueError("No '+'-prefixed args found in job args.")
    name = "_".join(name_parts)
    if sid_part is not None:
        name += f"_{sid_part}"
    return name
