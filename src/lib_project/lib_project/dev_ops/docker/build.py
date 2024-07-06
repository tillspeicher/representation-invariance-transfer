import argparse
import os
from pathlib import Path
from typing import Union

from ...config_defs import DockerBuildConfig, ProjectConfig
from ..utils import run_command
from .docker_utils import (
    CONTAINER_HOME_DIR,
    get_image_hash,
    get_image_name,
    get_image_type,
    get_logger,
    get_nfs_image_hash_path,
    get_nfs_image_path,
    prefix_list_elements,
)


"""
This script builds the project as a docker container
"""

logger = get_logger(__name__)

# BUILDER_BASE_IMAGE = "docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel"
# TARGET_BASE_IMAGES = {
#     "devel": BUILDER_BASE_IMAGE,
#     "runtime": "docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
# }
FILE_DIR = Path(__file__).parent


def add_build_docker_parser(subparser: argparse._SubParsersAction) -> None:
    parser = subparser.add_parser(
        "build-docker",
        help="Build the project as a Docker container",
    )
    parser.add_argument(
        "--devel",
        action="store_true",
        default=False,
        help="Build a development image",
        # choices=["devel", "runtime"],
        # default="runtime",
        # help="Type of the image to build",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Do not export the image to the NFS",
    )
    parser.set_defaults(func=build_image)


def build_image(
    config: ProjectConfig,
    args: argparse.Namespace,
    unknown_args: list[str],
) -> None:
    image_name = config.docker.image
    target_image_type = get_image_type(args.devel)

    target_image_name = get_image_name(config.docker.image, args.devel)
    print(f"Building image {target_image_name}")
    build_command = _get_build_command(
        target_image_name,
        target_image_type,
        app_name=image_name,
        build_config=config.docker.build,
    )
    if not run_command(build_command):
        logger.error("Failed to build the image")
        return
    if not args.no_export:
        save_command = _get_save_command(
            config.docker.image_storage_dir,
            target_image_name,
        )
        if not run_command(save_command):
            logger.error("Failed to save the image")
            return


def _get_build_command(
    target_image_name: str,
    target_image_type: str,
    # project_dockerfile: Union[str, Path],
    app_name: str,
    build_config: DockerBuildConfig,
) -> str:
    print("target image type", target_image_type)
    target_base_image = build_config.target_base_image[target_image_type]
    print("target base image", target_base_image)
    builder_target_name = f"{target_image_name}_custom"
    input_image_names = [
        build_config.builder_base_image,
        "exp_base",
        target_base_image,
    ]
    output_image_names = [
        "exp_base",
        builder_target_name,
        target_image_name,
    ]
    # First, install Python and pip in the builder and target images
    dockerfiles = [
        FILE_DIR / "Dockerfile.base",
        build_config.dockerfile,
        FILE_DIR / "Dockerfile.final",
    ]
    build_args = [
        {},
        {},
        {
            "BUILDER_IMAGE_NAME": builder_target_name,
            "APP_NAME": app_name,
            "ENTRYPOINT_DIR": str(FILE_DIR),
            "HOST_UID": os.getuid(),
            "HOST_GID": os.getgid(),
        },
    ]

    build_commands = []
    for (
        input_image_name,
        output_image_name,
        project_dockerfile,
        custom_build_args,
    ) in zip(
        input_image_names,
        output_image_names,
        dockerfiles,
        build_args,
    ):
        stage_build_args = {
            "EXP_DIR": str(CONTAINER_HOME_DIR),
            "INPUT_IMAGE_NAME": input_image_name,
            **custom_build_args,
        }
        stringified_build_args = prefix_list_elements(
            "--build-arg ",
            [f"{key}={value}" for key, value in stage_build_args.items()],
        )
        build_command = " ".join(
            [
                "docker",
                "build",
                *stringified_build_args,
                "-t",
                output_image_name,
                "-f",
                str(project_dockerfile),
                ".",
            ]
        )
        build_commands.append(f"echo 'Building image {output_image_name}'")
        build_commands.append(build_command)
        build_commands.append(f"echo 'Finished building {output_image_name}'")
    return " && ".join(build_commands)


def _get_save_command(
    image_storage_dir: Union[str, Path],
    image_name: str,
) -> str:
    image_storage_path = get_nfs_image_path(image_storage_dir, image_name)
    if not image_storage_path.parent.exists():
        logger.warning(
            "Storate directory for images does not exist at "
            f"{image_storage_path.parent}. Skipping saving the image."
        )
        return ""

    image_hash_path = get_nfs_image_hash_path(image_storage_dir, image_name)
    image_hash = get_image_hash(image_name)
    return " && ".join(
        [
            f"echo 'Saving image {image_name}'",
            f"echo '{image_hash}' > {image_hash_path}",
            f"docker save {image_name} > {image_storage_path}",
        ]
    )
