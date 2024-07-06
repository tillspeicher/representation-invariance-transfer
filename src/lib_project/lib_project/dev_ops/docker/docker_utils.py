import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Union


CONTAINER_HOME_DIR = Path("/home/exp")


def get_logger(file_name: str) -> logging.Logger:
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def get_image_type(devel: bool) -> str:
    return "devel" if devel else "runtime"


def get_image_name(image: str, devel: bool) -> str:
    image_type = get_image_type(devel)
    return f"{image}_{image_type}"


def get_nfs_image_path(
    image_storage_dir: Union[str, Path],
    image_name: str,
) -> Path:
    return _get_nfs_storage_path(image_storage_dir, image_name, "tar")


def get_nfs_image_hash_path(
    image_storage_dir: Union[str, Path],
    image_name: str,
) -> Path:
    return _get_nfs_storage_path(image_storage_dir, image_name, "sha256")


def _get_nfs_storage_path(
    image_storage_dir: Union[str, Path],
    image_name: str,
    file_ending: str,
) -> Path:
    return Path(image_storage_dir) / f"{image_name}.{file_ending}"


def prefix_list_elements(prefix: str, list_: list[str]) -> list[str]:
    prefixed_elments = []
    for element in list_:
        prefixed_elments.append(prefix)
        prefixed_elments.append(element)
    return prefixed_elments


def is_running_on_slurm() -> bool:
    return "SLURM_JOB_ID" in os.environ


def get_image_hash(image_name: str) -> str:
    cmd = f"docker inspect {image_name} --format '{{{{ .Id }}}}'"
    output = subprocess.check_output(cmd, shell=True, text=True).strip()
    return output


def get_gpu_ids(gpus: str) -> str:
    gpu_count = _get_cuda_gpu_count()
    if gpus:
        gpu_args = [int(gid) for gid in gpus.split(",")]
        if gpu_count == 0:
            raise ValueError("No GPUs found, but --gpus was specified.")
        if any(gpu_id >= gpu_count for gpu_id in gpu_args):
            raise ValueError("GPU ID out of range.")
        gpu_ids = ",".join([str(gpu) for gpu in gpu_args])
    else:
        gpu_ids = ",".join([str(i) for i in range(gpu_count)])
    return gpu_ids


def _get_cuda_gpu_count() -> int:
    try:
        command = "nvidia-smi --list-gpus"
        output = subprocess.check_output(command, shell=True, text=True)
        gpu_count = len(output.strip().split("\n"))
        return gpu_count
    except subprocess.CalledProcessError:
        return 0
