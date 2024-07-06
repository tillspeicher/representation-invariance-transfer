import logging
import os
from pathlib import Path
from typing import Optional

import paramiko
from hydra.core.config_store import ConfigStore

from lib_project.config_defs import ServerConfig, UploadConfig


def upload_files(
    source_files: list[str],
    target_files: list[str],
) -> None:
    if len(source_files) != len(target_files):
        raise ValueError(
            "The number of source files and target files does not match."
        )
    server_config, upload_config = get_upload_config()

    with paramiko.SSHClient() as ssh:
        ssh.load_host_keys(str(Path.home() / ".ssh/known_hosts"))
        try:
            ssh.connect(
                server_config.contact_address,
                username=server_config.username,
            )
        except paramiko.AuthenticationException as e:
            print("Error authenticating to server:", e)
            return

        results_root = Path(upload_config.results_root_dir)
        with ssh.open_sftp() as sftp:
            for src_file, target_file in zip(source_files, target_files):
                target_file = results_root / target_file
                # TODO: try to group file together by directories to avoid
                # mkdir calls on existing directories
                # remote_mkdir_recursive(sftp, file_dir, target_dir)
                # print(f"Uploading {src_file} to {target_file}")
                try:
                    sftp.put(src_file, str(target_file))
                except FileNotFoundError as e:
                    logging.error(
                        f"Error uploading {src_file} to {target_file}: {e}"
                    )
                    break

    logging.info(f"Uploaded {len(source_files)} files to server.")


# def remote_mkdir_recursive(sftp: paramiko.SFTP, directory: Path):
#     cur_dir = base_dir
#     for path_component in path_components:
#         cur_dir = os.path.join(cur_dir, path_component)
#         try:
#             sftp.mkdir(cur_dir)
#         except IOError:
#             # The directory already exists
#             pass


def register_upload_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="upload",
        node=UploadConfig,
    )


_server_config: Optional[ServerConfig] = None
_upload_config: Optional[UploadConfig] = None


def set_upload_config(
    server_config: ServerConfig,
    upload_config: UploadConfig,
) -> None:
    global _server_config, _upload_config
    _server_config = server_config
    _upload_config = upload_config


def get_upload_config(
    manual_server_config: Optional[ServerConfig] = None,
    manual_upload_config: Optional[UploadConfig] = None,
) -> tuple[ServerConfig, UploadConfig]:
    global _server_config, _upload_config
    server_config = (
        manual_server_config
        if manual_server_config is not None
        else _server_config
    )
    upload_config = (
        manual_upload_config
        if manual_upload_config is not None
        else _upload_config
    )
    if server_config is None:
        username = os.environ.get(ServerConfig.USERNAME_KEY)
        contact_address = os.environ.get(ServerConfig.CONTACT_ADDRESS_KEY)

        if username is None:
            raise ValueError(
                "Username for upload is not defined. "
                "Make sure you set the environment variable "
                f"'{ServerConfig.USERNAME_KEY}'"
            )
        if contact_address is None:
            raise ValueError(
                "Host address for upload is not defined. "
                "Make sure you set the environment variable "
                f"'{ServerConfig.CONTACT_ADDRESS_KEY}'"
            )
        _server_config = ServerConfig(
            username=username,
            contact_address=contact_address,
        )
        server_config = _server_config

    if upload_config is None:
        results_root_dir = os.environ.get(UploadConfig.RESULTS_ROOT_DIR_KEY)
        results_url_prefix = os.environ.get(UploadConfig.RESULTS_URL_PREFIX_KEY)

        if results_root_dir is None:
            raise ValueError(
                "Results root directory for upload is not defined. "
                "Make sure you set the environment variable "
                f"'{UploadConfig.RESULTS_ROOT_DIR_KEY}'"
            )
        if results_url_prefix is None:
            raise ValueError(
                "Results URL prefix for upload is not defined. "
                "Make sure you set the environment variable "
                f"'{UploadConfig.RESULTS_URL_PREFIX_KEY}'"
            )

        _upload_config = UploadConfig(
            results_root_dir=results_root_dir,
            results_url_prefix=results_url_prefix,
        )
        upload_config = _upload_config
    return server_config, upload_config
