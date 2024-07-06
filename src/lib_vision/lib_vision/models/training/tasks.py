import copy
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

import lightning as L
import pandas as pd
import torch

from lib_dl_base.defs.task_id import TaskID
from lib_vision.data import MulticlassDataModule

from ..architectures import ModelConfig
from .fine_tuning import (
    FineTuningFreezeConfig,
    copy_model,
    prepare_torchvision_finetuning,
)
from .objectives.supervised import ACCURACY_METRIC, SupervisedLearning
from .optimization import OptimizerConfig
from .train import TrainingConfig, TrainingResult
from .train import train as train_model
from .utils import extract_model


@dataclass
class EvalConfig:
    results_df: pd.DataFrame
    dataset_name: Optional[str] = None


@dataclass
class ModelInfo:
    name: str
    model: torch.nn.Module
    config: Optional[ModelConfig] = None


D = TypeVar("D", bound=L.LightningDataModule)


@dataclass
class DataInfo(Generic[D]):
    name: str
    data: D


def train_supervised(
    description: TaskID,
    model: ModelInfo,
    dataset: DataInfo[MulticlassDataModule],
    training_config: TrainingConfig,
    *,
    keep_training_wrapper: bool = False,
    optimizer_config: OptimizerConfig = OptimizerConfig(),
) -> TrainingResult:
    if keep_training_wrapper and isinstance(model, SupervisedLearning):
        task = model
    else:
        task = SupervisedLearning(
            model=model.model,
            classes=dataset.data.num_classes,
            test_metrics=[ACCURACY_METRIC],
            optimizer_config=optimizer_config,
        )

    training_res = train_model(
        (
            description.set_action("supervised_training")
            .set_model(model.name)
            .set_dataset(dataset.name)
        ),
        data=dataset.data,
        config=training_config,
        training_task=task,
        eval_task=task,
    )
    if keep_training_wrapper:
        supervised_res = training_res
    else:
        supervised_res = extract_model(training_res)
    return supervised_res


def fine_tune_supervised(
    description: TaskID,
    model: ModelInfo,
    dataset: DataInfo[MulticlassDataModule],
    training_config: TrainingConfig,
    freeze_config: FineTuningFreezeConfig,
    optimizer_config: OptimizerConfig = OptimizerConfig(),
) -> TrainingResult:
    if model.config is None:
        raise ValueError("Model config must be provided for fine-tuning.")
    ft_model_config = copy.deepcopy(model.config)
    ft_model_config.num_classes = dataset.data.num_classes
    fine_tuning_model = prepare_torchvision_finetuning(
        copy_model(model.model, model.config),
        ft_model_config,
        freeze_config=freeze_config,
    )
    task = SupervisedLearning(
        model=fine_tuning_model,
        classes=dataset.data.num_classes,
        test_metrics=[ACCURACY_METRIC],
        optimizer_config=optimizer_config,
    )
    fine_tuning_result = extract_model(
        train_model(
            (
                description.set_action("fine_tuning")
                .set_model(model.name)
                .set_dataset(dataset.name)
            ),
            data=dataset.data,
            config=training_config,
            training_task=task,
            eval_task=task,
        )
    )
    return fine_tuning_result
