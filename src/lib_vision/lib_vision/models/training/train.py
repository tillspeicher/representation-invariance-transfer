import os
import time
from dataclasses import dataclass, field
from typing import Optional

import lightning as L
import pandas as pd
import wandb
from lightning.pytorch.loggers import Logger, WandbLogger

from lib_dl_base.defs.task_id import TaskID
from lib_dl_base.io import persistence as results_persistence

from ..io.logging import delete_prev_logs, get_csv_logger
from .persistence import (
    TrainingMetadata,
    TrainingResult,
    load_training_result,
    save_training_metadata,
)
from .trainer import DeviceConfig, Trainer


EVAL_RESULT_NAME = "eval_metrics"


@dataclass
class TrainingConfig:
    max_epochs: int
    save_checkpoints: bool
    train: bool = True
    eval: bool = True
    devices: DeviceConfig = field(default_factory=DeviceConfig)
    max_steps: int = -1
    log_every_n_steps: int | None = None
    wandb_project_name: str | None = None


def train(
    task_id: TaskID,
    data: L.LightningDataModule,
    config: TrainingConfig,
    training_task: Optional[L.LightningModule] = None,
    eval_task: Optional[L.LightningModule] = None,
) -> TrainingResult:
    assert training_task is not None or eval_task is not None
    if training_task is not None:
        if config.train:
            training_result = _train_model(
                task_id,
                data,
                config,
                training_task,
            )
        else:
            print(f"Loading previous training result for model {task_id.name}")
            training_result = load_training_result(
                task_id,
                training_task,
            )
    else:
        training_result = TrainingResult(
            model=None,
            metadata=TrainingMetadata(
                task_id=task_id,
                training_time=0,
            ),
        )

    if eval_task is not None:
        if config.eval:
            eval_result = _eval_on_task(
                task_id,
                data,
                config,
                eval_task,
            )
        else:
            print(f"Loading previous eval result for model {task_id.name}")
            eval_result = results_persistence.load(
                task_id,
                file_name=EVAL_RESULT_NAME,
                obj_type=pd.DataFrame,
                # f"{model_name}_{TEST_RESULT_POSTFIX}",
            )
        training_result.metrics = eval_result
    return training_result


def _train_model(
    task_id: TaskID,
    data: L.LightningDataModule,
    config: TrainingConfig,
    task: L.LightningModule,
) -> TrainingResult:
    print(
        "--------------\n" f"Training model {task_id.name} \n" "--------------"
    )
    delete_prev_logs(task_id)

    wandb_run = None
    if config.log_every_n_steps is not None:
        loggers: list[Logger] = [get_csv_logger(task_id)]
        if config.wandb_project_name is not None:
            wandb_run = wandb.init(
                project=config.wandb_project_name,
                name=task_id.name,
                reinit=True,
            )
            loggers.append(
                WandbLogger(
                    log_model="all",
                    project=config.wandb_project_name,
                    name=task_id.name,
                )
            )
    else:
        loggers = []

    start_time = time.perf_counter()
    trainer = Trainer(
        task_id=task_id,
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        limit_train_batches=1.0 if config.max_steps < 0 else config.max_steps,
        limit_val_batches=1.0 if config.max_steps < 0 else config.max_steps,
        devices=config.devices,
        enable_checkpointing=config.save_checkpoints,
        enable_progress_bar=should_show_progress_bar(),
        logger=loggers,
        log_every_n_steps=config.log_every_n_steps,
    )
    trainer.fit(task, data)
    end_time = time.perf_counter()
    training_time = end_time - start_time

    if wandb_run is not None:
        wandb_run.finish()

    metadata = TrainingMetadata(
        task_id=task_id,
        training_time=training_time,
    )
    save_training_metadata(metadata)
    result = TrainingResult(
        metadata=metadata,
        model=task,
    )
    return result


def _eval_on_task(
    task_id: TaskID,
    data: L.LightningDataModule,
    config: TrainingConfig,
    task: L.LightningModule,
) -> pd.DataFrame:
    assert hasattr(task, "test_metrics")
    print(
        "--------------\n" f"Evaluating model {task_id.name}\n" "--------------"
    )
    trainer = Trainer(
        task_id=task_id,
        # 'max_steps' is used for debugging to just test whether the
        # respective loops work. Therefore, we also pass it to
        # the eval function.
        max_steps=config.max_steps,
        limit_test_batches=1.0 if config.max_steps < 0 else config.max_steps,
        devices=config.devices,
        enable_checkpointing=False,
        enable_progress_bar=should_show_progress_bar(),
    )
    for split in ["train", "test"]:
        data.setup(split)
    data_split_results = {}
    for data_split, dataloader in zip(
        ["train", "val", "test"],
        [
            data.train_dataloader(),
            data.val_dataloader(),
            data.test_dataloader(),
        ],
    ):
        trainer.test(task, dataloaders=[dataloader])
        eval_res = task.test_metrics.compute()  # type: ignore
        data_split_results[data_split] = [data_split] + [
            metric_res.item() for metric_res in eval_res.values()
        ]
    result = pd.DataFrame.from_dict(
        data_split_results,
        orient="index",
        columns=["split"] + list(task.test_metrics.keys()),
    ).set_index("split")
    results_persistence.save(
        result,
        task_id,
        file_name=EVAL_RESULT_NAME,
    )
    return result


DISABLE_PROGRESSBAR_KEY = "DISABLE_PROGRESSBAR"


def should_show_progress_bar() -> bool:
    if (
        DISABLE_PROGRESSBAR_KEY in os.environ
        and os.environ[DISABLE_PROGRESSBAR_KEY] == "1"
    ):
        return False
    return True
