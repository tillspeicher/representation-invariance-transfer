import os
import shutil
from pathlib import Path
from typing import Literal, Optional, Union

from ..defs.task_id import TaskID


ARTIFACTS_DIR_KEY = "ARTIFACTS_DIR"
_artifacts_dir: Optional[Path] = None


def set_artifacts_dir(dir: Union[str, Path]) -> None:
    global _artifacts_dir
    _artifacts_dir = Path(dir)
    os.environ[ARTIFACTS_DIR_KEY] = str(_artifacts_dir)


def get_artifacts_dir() -> Path:
    global _artifacts_dir
    if _artifacts_dir is not None:
        return _artifacts_dir
    else:
        if ARTIFACTS_DIR_KEY in os.environ:
            set_artifacts_dir(os.environ[ARTIFACTS_DIR_KEY])
        if _artifacts_dir is not None:
            return _artifacts_dir
        raise ValueError(
            "Artifacts directory not set."
            f"Either set an environment variable named {ARTIFACTS_DIR_KEY} "
            "to point to the directory where artifacts "
            "(training logs and checkpoints) should be located "
            "or pass it as a Hydra configuration option under "
            "'dirs.artifacts'."
        )


def append_directories(
    base_dir: Path,
    task_id: TaskID,
) -> Path:
    return base_dir / task_id.path


def get_logging_dir(
    task_id: TaskID,
    log_type: Literal["tb", "csv"],
) -> Path:
    return append_directories(get_artifacts_dir(), task_id) / f"{log_type}_logs"


def get_checkpoints_dir(
    task_id: TaskID,
) -> Path:
    return append_directories(get_artifacts_dir(), task_id) / "checkpoints"


def recreate_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


DATA_DIR_KEY = "DATA_DIR"
_dataset_dir: Optional[Path] = None


def set_dataset_dir(dir: Union[str, Path]) -> None:
    global _dataset_dir
    _dataset_dir = Path(dir)
    os.environ[DATA_DIR_KEY] = str(_dataset_dir)


def _get_dataset_dir() -> Path:
    global _dataset_dir
    if _dataset_dir is not None:
        return _dataset_dir
    else:
        if DATA_DIR_KEY in os.environ:
            set_dataset_dir(os.environ[DATA_DIR_KEY])
        if _dataset_dir is not None:
            return _dataset_dir
        raise ValueError(
            "Dataset directory not set."
            f"Either set an environment variable named {DATA_DIR_KEY} "
            "to point to the directory where dataset files "
            "should be located "
            "or pass it as a Hydra configuration option under 'dirs.data'."
        )


def get_dataset_dir(
    dataset_name: str,
) -> Path:
    return _get_dataset_dir() / dataset_name


def rmdir_recursive(dir: Path) -> None:
    if dir.is_dir():
        shutil.rmtree(dir)
