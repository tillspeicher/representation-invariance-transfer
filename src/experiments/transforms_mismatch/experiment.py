from copy import deepcopy
from dataclasses import dataclass
from typing import cast

import pandas as pd

from lib_project.experiment import ExperimentConfig as BaseExperimentConfig
from lib_project.experiment import ExperimentID, experiment
from lib_vision.models.architectures import ModelConfig, create_model
from lib_vision.models.training import (
    FineTuningFreezeConfig,
    ModelInfo,
    TrainingConfig,
    fine_tune_supervised,
    multi_training,
    train_supervised,
)

from .data import TMDataConfig, create_tm_dataset


EXP_NAME = "transforms_mismatch"
EXP_ABBREVIATION = "tm"


@dataclass
class ExperimentConfig(BaseExperimentConfig):
    training: TrainingConfig
    fine_tuning_training: TrainingConfig
    fine_tuning_freeze: FineTuningFreezeConfig
    data: TMDataConfig
    model: ModelConfig


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    transforms: list[str]
    in_dist_performance: pd.DataFrame
    transfer_performance: pd.DataFrame


@experiment(EXP_NAME)
def tm_experiment(
    config: ExperimentConfig,
    experiment_id: ExperimentID,
) -> ExperimentResult:
    data = create_tm_dataset(config.data)
    datasets = data.data

    models = [
        ModelInfo(ds_name, create_model(config.model), deepcopy(config.model))
        for ds_name in datasets.keys()
    ]

    training_res = multi_training(
        train_supervised,
        experiment_id,
        models=models,
        datasets=datasets,
        training_config=config.training,
    )
    trained_models = [
        ModelInfo(cast(str, model_name), model, deepcopy(config.model))
        for model_name, model in training_res.models("model").items()
    ]

    transfer_res = multi_training(
        fine_tune_supervised,
        experiment_id,
        models=trained_models,
        datasets=datasets,
        training_config=config.fine_tuning_training,
        cross_product=True,
        freeze_config=config.fine_tuning_freeze,
    )

    return ExperimentResult(
        config=config,
        transforms=data.transforms,
        in_dist_performance=training_res.metrics,
        transfer_performance=transfer_res.metrics,
    )
