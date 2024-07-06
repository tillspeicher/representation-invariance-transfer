import itertools
import os
import subprocess
from dataclasses import dataclass
from typing import Iterator, Optional

import paramiko


@dataclass
class TaskConfig:
    experiment: str
    config_names: list[str]
    seed_ids: list[int]
    experiment_options: list[str]
    other_options: Optional[str] = None


@dataclass
class MachineConfig:
    name: str
    device_ids: list[int]


@dataclass
class EnvConfig:
    username: str
    experiment_dir: str
    path_expansion: str


def run(
    env_config: EnvConfig,
    machines: list[MachineConfig],
    task_config: TaskConfig,
):
    # Copy the source files first
    # subprocess.run(["./sync.sh", "to-svr", "src/"])
    subprocess.run(["./scripts/sync.sh", "to-svr", "src/"])

    commands = get_commands(task_config)
    commands_exhausted = False
    for machine in itertools.cycle(machines):
        machine_commands = []
        for device_id in machine.device_ids:
            try:
                # command = f"CUDA_VISIBLE_DEVICES={device_id} {next(commands)}"
                command = next(commands).format(gpus=device_id)
                machine_commands.append(command)
            except StopIteration:
                # No more commands to run
                commands_exhausted = True
                break
        exec_commands_on_machine(
            env_config,
            machine.name,
            machine_commands,
        )
        if commands_exhausted:
            break

    try:
        missed_command = next(commands)
        raise ValueError(
            f"Not all commands were executed, missed {missed_command}"
        )
    except StopIteration:
        pass


def get_commands(
    task_config: TaskConfig,
) -> Iterator[str]:
    experiment_options_str = " ".join(
        [
            f"++{task_config.experiment}.{option}"
            for option in task_config.experiment_options
        ]
    )
    other_options_str = (
        f" {task_config.other_options}" if task_config.other_options else ""
    )
    options_str = f"{experiment_options_str}{other_options_str}"
    for config_name in task_config.config_names:
        for seed_id in task_config.seed_ids:
            yield (
                # "DISABLE_PROGRESSBAR=1 "
                # f"nohup poetry run python3 -u src/main.py +sid={seed_id} "
                "nohup ./scripts/run_docker.py --gpus={gpus} --no-interaction "
                "--no-progressbar "  # Gets substituted later
                # "python -u src/main.py "
                "deepspeed src/main.py "
                f"+sid={seed_id} "
                f"+{task_config.experiment}={config_name} {options_str} "
                f"> logs/{task_config.experiment}_{config_name}_{seed_id}.out "
                "2>&1 &"  # Redirect stderr to stdout
            )


def exec_commands_on_machine(
    env_config: EnvConfig,
    machine_name: str,
    commands: list[str],
) -> Optional[tuple]:
    stdout = None
    stderr = None
    with paramiko.SSHClient() as con:
        con.load_host_keys(
            os.path.expanduser(os.path.join("~", ".ssh", "known_hosts"))
        )
        try:
            con.connect(
                machine_name,
                username=env_config.username,
            )
            print(f"Logged into {machine_name}")
        except paramiko.AuthenticationException:
            # print(f"Error authenticating to server '{machine_name}'")
            raise paramiko.AuthenticationException(
                f"Error authenticating to server '{machine_name}'"
            )

        # Change the working directory to that of the project
        combined_commands = (
            f"cd {env_config.experiment_dir};\n"
            + f"export PATH={env_config.path_expansion}:$PATH;\n"
            + ";\n".join(commands)
            + "; exit;"
        )
        print("executing:", combined_commands)
        try:
            stdin, stdout, stderr = con.exec_command(
                combined_commands,
                timeout=8,
            )
            print_output(stderr, "Error: ")
            print_output(stdout)
            return stdout, stderr
        except TimeoutError:
            # TODO: the timeout trick is hacky, but nohup commands
            # won't terminate otherwise
            pass


def print_output(stdstream, prefix: str = "") -> None:
    stdstream.channel.recv_exit_status()
    for line in stdstream.readlines():
        print(f"{prefix}{line.strip()}")
