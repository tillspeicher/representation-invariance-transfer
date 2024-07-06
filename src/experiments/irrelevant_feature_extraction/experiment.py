from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from lib_project.experiment import ExperimentConfig as BaseExperimentConfig
from lib_project.experiment import ExperimentID, experiment
from lib_vision.models.architectures import ModelConfig, create_model
from lib_vision.models.metrics.representations.invariance import (
    InvarianceMeasurementConfig,
    multi_measure_invariance,
)
from lib_vision.models.training import (
    FineTuningFreezeConfig,
    ModelInfo,
    TrainingConfig,
    TrainingMetadata,
    fine_tune_supervised,
    multi_training,
    train_supervised,
)

from .data import (
    MIXED_CIFAR_AVAILABILITY_KEY,
    MIXED_CIFAR_COR_KEY,
    MIXED_OBJECTS_KEY,
    OBJECTS_ONLY_KEY,
    IFEDataConfig,
    create_invariance_measurement_datasets,
    create_irrelevant_features_dataset,
)


EXP_NAME = "irrelevant_feature_extraction"
EXP_ABBREVIATION = "ife"


@dataclass
class ExperimentConfig(BaseExperimentConfig):
    training: TrainingConfig
    fine_tuning: TrainingConfig
    data: IFEDataConfig
    model: ModelConfig
    transforms: list[str]
    compute_invariance: bool = True
    all_models: bool = False


@dataclass
class ExperimentResult:
    objects: list[int]
    training_performance: pd.DataFrame
    training_metadata: dict[str, TrainingMetadata]
    invariances: Optional[dict[str, pd.DataFrame]]
    core_ft_performance: pd.DataFrame
    core_ft_metadata: dict[str, TrainingMetadata]
    extended_ft_performance: Optional[pd.DataFrame]
    extended_ft_metadata: Optional[dict[str, TrainingMetadata]]


CIFAR_ONLY_KEY = "cifar_only"
MIXED_CIFAR_KEY = "cifar_mixed"
CIFAR_VS_NO_CIFAR_COL = "cifar_vs_no_cifar"
PATCH_VS_NO_PATCH_COL = "patch_vs_no_patch"


@experiment(EXP_NAME)
def ife_experiment(
    config: ExperimentConfig,
    task_id: ExperimentID,
) -> ExperimentResult:
    irrelevant_features_data = create_irrelevant_features_dataset(
        config.data,
        transform_names=config.transforms,
    )
    training_datasets = irrelevant_features_data.datasets

    core_ds_names = {
        CIFAR_ONLY_KEY: f"{MIXED_CIFAR_AVAILABILITY_KEY}0.0",
        MIXED_CIFAR_KEY: f"{MIXED_CIFAR_COR_KEY}0.0",
        MIXED_OBJECTS_KEY: MIXED_OBJECTS_KEY,
        OBJECTS_ONLY_KEY: OBJECTS_ONLY_KEY,
    }
    if not config.all_models:
        training_datasets = {
            ds_name: dataset
            for ds_name, dataset in training_datasets.items()
            if ds_name in core_ds_names.values()
        }

    models = []
    dataset_classes = {}
    for data_name in training_datasets.keys():
        model_config = deepcopy(config.model)
        if data_name.startswith("cifar"):
            num_classes = 10
        else:
            num_classes = config.data.n_classes
        model_name = data_name
        dataset_classes[data_name] = num_classes
        # models[model_name] = create_model(model_config)
        models.append(
            ModelInfo(model_name, create_model(model_config), model_config)
        )
    training_res = multi_training(
        train_supervised,
        task_id,
        models=models,
        datasets=training_datasets,
        training_config=config.training,
    )
    trained_models = [
        ModelInfo(
            target_name, training_res.models("model")[ds_name], config.model
        )
        for target_name, ds_name in core_ds_names.items()
    ]

    if config.compute_invariance:
        invariances = compute_invariance(
            task_id,
            trained_models,
            config.data,
            objects=irrelevant_features_data.objects,
        )
        print("invariances:", invariances)
    else:
        invariances = None

    core_transfer_datasets = {
        target_name: training_datasets[ds_name]
        for target_name, ds_name in core_ds_names.items()
    }
    core_ft_res = multi_training(
        fine_tune_supervised,
        task_id,
        models=trained_models,
        datasets=core_transfer_datasets,
        training_config=config.fine_tuning,
        cross_product=True,
        freeze_config=FineTuningFreezeConfig("linear_probe"),
    )

    if config.all_models:
        extended_ft_models = [
            ModelInfo(
                ds_name, training_res.models("model")[ds_name], config.model
            )
            for ds_name in training_datasets.keys()
            if ds_name not in core_ds_names.values()
        ]
        extended_ft_target_datasets = {
            CIFAR_ONLY_KEY: core_transfer_datasets[CIFAR_ONLY_KEY],
            OBJECTS_ONLY_KEY: core_transfer_datasets[OBJECTS_ONLY_KEY],
        }
        extended_ft_res = multi_training(
            fine_tune_supervised,
            task_id,
            models=extended_ft_models,
            datasets=extended_ft_target_datasets,
            training_config=config.fine_tuning,
            cross_product=True,
            freeze_config=FineTuningFreezeConfig("linear_probe"),
        )
        extended_ft_performance = extended_ft_res.metrics
        extended_ft_metadata = {
            "--".join(model_key): metadata
            for model_key, metadata in extended_ft_res.metadata()
        }
    else:
        extended_ft_performance = None
        extended_ft_metadata = None

    core_ft_metadata = {
        "--".join(model_key): metadata
        for model_key, metadata in core_ft_res.metadata().items()
    }
    result = ExperimentResult(
        objects=irrelevant_features_data.objects,
        training_performance=training_res.metrics,
        training_metadata=training_res.metadata("model"),
        invariances=invariances,
        core_ft_performance=core_ft_res.metrics,
        core_ft_metadata=core_ft_metadata,
        extended_ft_performance=extended_ft_performance,
        extended_ft_metadata=extended_ft_metadata,
    )
    return result


def compute_invariance(
    task_id: ExperimentID,
    model_infos: list[ModelInfo],
    data_config: IFEDataConfig,
    objects: list[int],
) -> dict[str, pd.DataFrame]:
    invariance_estimation_data = create_invariance_measurement_datasets(
        data_config,
        objects,
    )
    datasets_1 = {
        variation_type: data[0]
        for variation_type, data in invariance_estimation_data.items()
    }
    datasets_2 = {
        variation_type: data[1]
        for variation_type, data in invariance_estimation_data.items()
    }
    models = {mi.name: mi.model for mi in model_infos}
    model_configs = {mi.name: mi.config for mi in model_infos}
    invariance_config = InvarianceMeasurementConfig(
        metrics=["l2", "cos"],
        shuffle_seed=data_config.sampling_seed + 39,
    )

    invariances = multi_measure_invariance(
        task_id=task_id,
        config=invariance_config,
        models=models,
        model_configs=model_configs,
        datasets_1=datasets_1,
        datasets_2=datasets_2,
    )
    return invariances
