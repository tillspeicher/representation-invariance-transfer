from pathlib import Path
from typing import Any, Optional

import lightning as L
import torch
from lightning.pytorch import callbacks as pl_callbacks

from lib_dl_base.defs.task_id import TaskID
from lib_dl_base.io.dirs import (
    get_checkpoints_dir,
    recreate_dir,
    rmdir_recursive,
)

from ...training.objectives import SupervisedLearning


def get_checkpoints_callback(
    description: TaskID,
    save_interval: Optional[int] = None,
) -> pl_callbacks.ModelCheckpoint:
    checkpoints_dir = get_checkpoints_dir(description)
    print("Deleting old checkpoings at", checkpoints_dir)
    rmdir_recursive(checkpoints_dir)
    recreate_dir(checkpoints_dir)

    return pl_callbacks.ModelCheckpoint(
        dirpath=checkpoints_dir,
        every_n_epochs=save_interval,
        # monitor="loss/val",
        # save_top_k=1,
        # filename="ckpt-{epoch:02d}-{global_step}-{val_loss:.2f}",
        # save_last=True,
    )


def get_best_checkpoint_path(
    description: TaskID,
) -> Path:
    checkpoints_dir = get_checkpoints_dir(description)
    if not checkpoints_dir.is_dir():
        raise ValueError(
            f"'{checkpoints_dir}' is not a valid (checkpoint) directory."
        )
    chkpt_files = sorted([f for f in checkpoints_dir.iterdir()])
    if len(chkpt_files) == 0:
        raise FileNotFoundError(f"No checkpoint found at {checkpoints_dir}.")
    # assert len(chkpt_files) == 1, "There should only be 1 checkpoint per task."
    # Load the best-performing checkpoint
    return chkpt_files[-1]


def load_model(
    description: TaskID,
    task: L.LightningModule,
) -> torch.nn.Module:
    best_checkpoint = get_best_checkpoint_path(description)
    if isinstance(task, SupervisedLearning):
        constructor_args: dict[str, Any] = {
            "model": task.model,
        }
    # elif isinstance(task, RepSimLearning):
    #     constructor_args = {
    #         "model_1": task.model_1.model,
    #         "model_2": task.model_2.model,
    #     }
    # elif isinstance(task, KnowledgeDistillation):
    #     constructor_args = {
    #         "teacher_model": task.teacher_model,
    #         "student_model": task.student_model,
    #     }
    else:
        raise ValueError(f"Unknown task type: {type(task)}")

    return type(task).load_from_checkpoint(best_checkpoint, **constructor_args)
