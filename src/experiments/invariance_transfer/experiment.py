from dataclasses import dataclass
from typing import Optional

import pandas as pd

from lib_project.experiment import ExperimentConfig as BaseExperimentConfig
from lib_project.experiment import ExperimentID, experiment
from lib_vision.data.lib.multiclass_datamodule import MulticlassDataModule
from lib_vision.models.architectures import ModelConfig, create_model
from lib_vision.models.training import (
    ModelInfo,
    MultiTrainingResult,
    TrainingConfig,
    TrainingMetadata,
    fine_tune_supervised,
    multi_training,
    train_supervised,
)
from utils.invariance_measurement import (
    InvarianceMeasurementConfig,
    STDataConfig,
    create_single_transform_datasets,
    measure_single_transform_invariance,
)


EXP_NAME = "invariance_transfer"
EXP_ABBREVIATION = "it"


@dataclass
class SeedConfig:
    config_seed: int
    sampling_seed: int
    transforms_sampling_seeds: tuple[int, int]


@dataclass
class ExperimentConfig(BaseExperimentConfig):
    seed: SeedConfig
    training: TrainingConfig
    fine_tuning: TrainingConfig
    model: ModelConfig
    training_data: STDataConfig
    eval_data: STDataConfig
    inv_measurement: InvarianceMeasurementConfig
    fine_tune: bool = False


@dataclass
class ExperimentResult:
    training_performance: pd.DataFrame
    training_metadata: dict[str, TrainingMetadata]
    invariances: dict[str, dict[str, pd.DataFrame]]
    ft_performance: Optional[pd.DataFrame]
    ft_metadata: Optional[dict[tuple[str, str], TrainingMetadata]]


@experiment(EXP_NAME)
def it_experiment(
    config: ExperimentConfig,
    experiment_id: ExperimentID,
) -> ExperimentResult:
    datasets = create_single_transform_datasets(
        config.training_data,
        include_none=True,
    )
    training_datasets = datasets["op-img"] | {
        f"{transform}_rand": dataset
        for transform, dataset in datasets["rp-rand"].items()
    }

    training_res = _train_models(experiment_id, config, training_datasets)
    trained_models = training_res.models("model")
    # Add an untrained model as reference for the default invariance
    # that models can achieve without training
    trained_models["untrained"] = create_model(config.model)

    invariances = measure_single_transform_invariance(
        experiment_id,
        config.inv_measurement,
        models=trained_models,
        model_config=config.model,
        data_config=config.eval_data,
        transforms_sampling_seeds=config.seed.transforms_sampling_seeds,
        use_rw_data="both",
    )

    if config.fine_tune:
        trained_model_infos = [
            ModelInfo(model=model, name=model_name, config=config.model)
            for model_name, model in trained_models.items()
        ]
        ft_res = multi_training(
            fine_tune_supervised,
            experiment_id,
            models=trained_model_infos,
            datasets=datasets,
            training_config=config.fine_tuning,
            cross_product=True,
        )
        ft_metrics = ft_res.metrics
        ft_metadata = ft_res.metadata()
    else:
        ft_metrics = None
        ft_metadata = None

    result = ExperimentResult(
        training_performance=training_res.metrics,
        training_metadata=training_res.metadata("model"),
        invariances=invariances,
        ft_performance=ft_metrics,
        ft_metadata=ft_metadata,
    )
    return result


def _train_models(
    experiment_id: ExperimentID,
    config: ExperimentConfig,
    datasets: dict[str, MulticlassDataModule],
) -> MultiTrainingResult:
    training_datasets = datasets
    models = [
        ModelInfo(
            model=create_model(config.model),
            name=transform_name,
        )
        for transform_name in training_datasets.keys()
    ]
    training_res = multi_training(
        train_supervised,
        experiment_id,
        models,
        training_datasets,
        training_config=config.training,
    )
    return training_res
