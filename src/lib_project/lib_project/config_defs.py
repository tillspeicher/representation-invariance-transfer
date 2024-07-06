from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Union


@dataclass
class DirectoryConfig:
    artifacts: Union[str, Path] = "artifacts"
    data: Union[str, Path] = "data"


@dataclass
class PortsConfig:
    jupyter: int = 8888

    def as_list(self) -> list[int]:
        return [self.jupyter]


@dataclass
class ServerConfig:
    username: str
    contact_address: str = "<server_url>"

    USERNAME_KEY: ClassVar[str] = "SERVER_USERNAME"
    CONTACT_ADDRESS_KEY: ClassVar[str] = "SERVER_CONTACT_ADDRESS"

    def to_env_vars(self) -> dict[str, str]:
        return {
            self.USERNAME_KEY: self.username,
            self.CONTACT_ADDRESS_KEY: self.contact_address,
        }


@dataclass
class UploadConfig:
    """Config for uploading results to a server.

    Attributes:
    results_root_dir: The root directory on the server to which
        the results will be uploaded.
    results_url_prefix: The URL prefix under which the resutls will be
        publicly available.
    """

    results_root_dir: str
    results_url_prefix: str

    RESULTS_ROOT_DIR_KEY: ClassVar[str] = "UPLOAD_ROOT_DIR"
    RESULTS_URL_PREFIX_KEY: ClassVar[str] = "RESULTS_URL_PREFIX"

    def to_env_vars(self) -> dict[str, str]:
        return {
            self.RESULTS_ROOT_DIR_KEY: self.results_root_dir,
            self.RESULTS_URL_PREFIX_KEY: self.results_url_prefix,
        }

    def init_results_root_dir(
        self,
        username: str,
        project_name: str,
    ) -> None:
        self.results_root_dir = f"/home/{username}/public_html/results/{project_name}"

    def init_results_url_prefix(
        self,
        username: str,
        project_name: str,
    ) -> None:
        self.results_url_prefix = f"<server_url>/{username}/results/{project_name}"


@dataclass
class SyncConfig:
    server_project_root: Union[str, Path]
    local_project_root: Union[str, Path] = "."


@dataclass
class DockerBuildConfig:
    dockerfile: Union[str, Path]
    builder_base_image: str = "docker.io/pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel"
    target_base_image: dict[str, str] = field(
        default_factory=lambda: {
            "runtime": "docker.io/pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime",
            "devel": "docker.io/pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel",
        }
    )


@dataclass
class DockerConfig:
    image: str
    build: DockerBuildConfig
    image_storage_dir: Union[str, Path]
    volumes: list[str] = field(default_factory=list)
    environment: list[str] = field(default_factory=list)
    env_file: list[Union[str, Path]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, config: dict) -> "DockerConfig":
        return cls(
            build=DockerBuildConfig(**config["build"]),
            **{k: v for k, v in config.items() if k != "build"},
        )


@dataclass
class ProjectConfig:
    dirs: DirectoryConfig
    ports: PortsConfig
    server: ServerConfig
    upload: UploadConfig
    sync: SyncConfig
    docker: DockerConfig
    sid: int = 0

    @classmethod
    def from_dict(cls, config: dict) -> "ProjectConfig":
        return cls(
            dirs=DirectoryConfig(**config["dirs"]),
            ports=PortsConfig(**config["ports"]),
            server=ServerConfig(**config["server"]),
            upload=UploadConfig(**config["upload"]),
            sync=SyncConfig(**config["sync"]),
            docker=DockerConfig.from_dict(config["docker"]),
            sid=config["sid"] if "sid" in config else 0,
        )
