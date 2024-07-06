import argparse
import subprocess

from ..config_defs import ProjectConfig


def add_connect_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        "connect", help="Establish an SSH connection with port forwarding."
    )
    parser.add_argument(
        "server",
        help="Server name",
    )
    parser.add_argument(
        "--port",
        type=int,
        help=(
            "Overwrite which port to forward. By default the port is read "
            "from the project config file."
        ),
    )
    parser.set_defaults(func=connect)


def connect(
    config: ProjectConfig,
    args: argparse.Namespace,
    unknown_args: list[str],
) -> None:
    server_config = config.server

    port = config.ports.jupyter
    if args.port is not None:
        port = args.port
    establish_ssh_connection(server_config.username, args.server, port)

    # Wait for user input to terminate the connection
    print("Press ENTER to terminate the SSH connection...")
    input()

    terminate_ssh_connection(server_config.username, args.server)


def establish_ssh_connection(username: str, server: str, port: int):
    cmd = [
        "ssh",
        "-f",
        "-N",
        "-M",
        "-S",
        "~/.ssh/control-%C",
        "-L",
        f"{port}:localhost:{port}",
        f"{username}@{server}",
    ]
    subprocess.run(cmd)
    print(
        "SSH connection established with port forwarding from "
        f"{server}:{port} to localhost:{port}"
    )


def terminate_ssh_connection(username, server):
    cmd = [
        "ssh",
        "-S",
        "~/.ssh/control-%C",
        "-O",
        "exit",
        f"{username}@{server}",
    ]
    subprocess.run(cmd)
    print("SSH connection terminated.")
