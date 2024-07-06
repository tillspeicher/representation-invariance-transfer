import argparse

# import sys
from pathlib import Path

from ..config_defs import ProjectConfig
from .utils import run_command


FILE_DIR = Path(__file__).parent
RSYNC_EXCLUDE_LIST = FILE_DIR / "rsync_exclude_list.txt"


# SyncDirection = Literal["to", "from"]


def add_sync_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "sync",
        help="Sync files with server",
    )
    parser.add_argument(
        "direction",
        choices=["to", "from"],
        help="Direction of sync",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Source and destination files",
    )
    parser.set_defaults(func=sync_files)
    # parser.add_argument(
    #     "--dry-run",
    #     action="store_true",
    #     help="Dry run (don't make changes)",
    # )


# def sync(server_dir: str | Path, workspace_dir: str | Path) -> None:
def sync_files(
    config: ProjectConfig,
    args: argparse.Namespace,
    unknown_args: list[str],
) -> None:
    server_config = config.server
    dir_config = config.sync
    workspace_dir = Path(dir_config.local_project_root)
    server_dir = Path(dir_config.server_project_root)
    # server_address = Path(
    #     f"{server_config.username}@{server_config.address}"
    #     f":(dir_config.server_project_root)"
    # )

    server_prefix = f"{server_config.username}@{server_config.contact_address}:"
    if args.direction == "to":
        src_base = workspace_dir
        src_prefix = ""
        dst_base = server_dir
        dst_prefix = server_prefix
    elif args.direction == "from":
        src_base = server_dir
        src_prefix = server_prefix
        dst_base = workspace_dir
        dst_prefix = ""
    else:
        print("Invalid direction.")
        return

    flags = f"-avu --delete --exclude-from={RSYNC_EXCLUDE_LIST}"
    # if args.dry_run:
    #     flags += " --dry-run"

    # print("Syncing with flags:", flags)
    # files = sys.argv[1:]
    # print("Syncing files:", files)
    for directory in args.files:
        src = src_base / directory
        dst = dst_base / directory
        if src.is_dir() or dst.is_dir():
            # Add trailing slash to directory paths, to avoid creating
            # a nested directory, i.e. to just copy the contents of the
            # directory.
            src = f"{src}/"
        else:
            src = str(src)
        src = f"{src_prefix}{src}"
        dst = f"{dst_prefix}{dst}"

        print(f"Syncing from {src} to {dst}")

        rsync_command = [
            "rsync",
            flags,
            "-e",
            "ssh",
            src,
            str(dst),
        ]
        run_command(" ".join(rsync_command))
