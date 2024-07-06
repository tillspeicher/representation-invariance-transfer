from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Mapping, Sequence, cast

import pandas as pd
import torch

from lib_project.experiment import ExperimentConfig as BaseExperimentConfig
from lib_project.experiment import ExperimentID, experiment
from lib_vision.data.lib.multiclass_datamodule import MulticlassDataModule
from lib_vision.models.architectures import ModelConfig, create_model
from lib_vision.models.training import (
    FineTuningFreezeConfig,
    ModelInfo,
    TrainingConfig,
    TrainingMetadata,
    fine_tune_supervised,
    multi_training,
    train_supervised,
)
from lib_vision.models.training.multi_training import ModelKey
from lib_vision.models.training.optimization import OptimizerConfig

from .data import SampledDataConfig, TvODataConfig, create_tvo_datasets


EXP_NAME = "transforms_vs_other"
EXP_ABBREVIATION = "tvo"

HYPERPARAMETER_LR_GRID = [
    1e-01,
    1e-02,
    1e-03,
    1e-04,
    1e-05,
]
FULL_FT_LEARNING_RATES = {
    "resnet-18": 1e-03,
    "resnet-50": 1e-03,
    "densenet-121": 1e-03,
    "vgg-11": 1e-03,
    "vit": 1e-03,
}


@dataclass
class ExperimentConfig(BaseExperimentConfig):
    training: TrainingConfig
    fine_tuning_training: TrainingConfig
    fine_tuning_freeze: FineTuningFreezeConfig
    data: TvODataConfig
    model: ModelConfig
    variation_category: str
    num_ft_samples: int | float = -1


@dataclass
class TransformRelationshipResult:
    training_performance: pd.DataFrame
    training_metadata: dict[str, TrainingMetadata]
    transfer_performance: pd.DataFrame
    transfer_metadata: dict[str, TrainingMetadata]


@dataclass
class ExperimentResult:
    sampled_config: SampledDataConfig
    same_transforms: TransformRelationshipResult
    disjoint_transforms: TransformRelationshipResult


@experiment(EXP_NAME)
def tvo_experiment(
    config: ExperimentConfig,
    experiment_id: ExperimentID,
) -> ExperimentResult:
    data = create_tvo_datasets(
        config.data,
        config.variation_category,
        config.num_ft_samples,
    )
    print("num_ft_samples", config.num_ft_samples)

    same_config = deepcopy(config)
    same_transforms_res = evaluate_transform_relationship(
        same_config,
        add_transform_rel_prefix(experiment_id, "same"),
        data.same_transforms_datasets,
        data.target_datasets,
    )
    disjoint_transforms_res = evaluate_transform_relationship(
        config,
        add_transform_rel_prefix(experiment_id, "disjoint"),
        data.disjoint_transforms_datasets,
        data.target_datasets,
    )
    result = ExperimentResult(
        sampled_config=data.sampled_config,
        same_transforms=same_transforms_res,
        disjoint_transforms=disjoint_transforms_res,
    )
    return result


def add_transform_rel_prefix(
    experiment_id: ExperimentID,
    transform_rel: str,
) -> ExperimentID:
    return cast(
        ExperimentID, experiment_id.add_prefix(f"{transform_rel}_transforms")
    )


def evaluate_transform_relationship(
    config: ExperimentConfig,
    experiment_id: ExperimentID,
    training_datasets: dict[str, MulticlassDataModule],
    target_datasets: dict[str, MulticlassDataModule],
) -> TransformRelationshipResult:
    models = create_models(
        config.model,
        config.variation_category,
        training_datasets,
    )

    finetune_whole_model = config.fine_tuning_freeze.type != "linear_probe"
    if finetune_whole_model:
        # Load pretrained models from another experiment
        # Disable training and evaluation
        config.training.train = False
        config.training.eval = False
        whole_model_ft_training_config = deepcopy(config.fine_tuning_training)
        config.fine_tuning_training.train = False
        config.fine_tuning_training.eval = False
        whole_model_ft_freeze_config = deepcopy(config.fine_tuning_freeze)
        config.fine_tuning_freeze.type = "linear_probe"
        config.fine_tuning_freeze.reset_head = True
        whole_model_ft_freeze_config.reset_head = False

        config_name = experiment_id.config_name
        assert isinstance(config_name, str)
        config_name_parts = config_name.split("_")
        ft_part = next(
            (part for part in config_name_parts if part.startswith("ft-")),
            None,
        )
        assert ft_part is not None, f"Could not find ft- part in {config_name}"
        ft_part_idx = config_name_parts.index(ft_part)
        source_model_config_name = "_".join(
            config_name_parts[:ft_part_idx] + ["ft-lp"]
        )
        source_model_experiment_id = deepcopy(experiment_id)
        source_model_experiment_id.config_name = source_model_config_name

        print(f"Source model experiment id: {source_model_experiment_id}")
    else:
        source_model_experiment_id = experiment_id
        whole_model_ft_training_config = None
        whole_model_ft_freeze_config = None

    training_res = multi_training(
        train_supervised,
        source_model_experiment_id,
        models=models,
        datasets=training_datasets,
        training_config=config.training,
    )

    trained_models = training_res.models("model")
    trained_model_infos, _ = _get_ft_model_infos(
        config,
        models,
        trained_models,
        target_datasets,
    )
    fine_tuning_res = multi_training(
        fine_tune_supervised,
        source_model_experiment_id,
        models=trained_model_infos,
        datasets=target_datasets,
        training_config=config.fine_tuning_training,
        cross_product=True,
        freeze_config=config.fine_tuning_freeze,
    )

    if whole_model_ft_training_config is not None:
        if config.variation_category == "finetuning_samples":
            # There is one pretrained model fine-tuned on several target
            # datasets here
            lp_trained_models = fine_tuning_res.models("dataset")
        else:
            # There are several pretrained models fine-tuned on one target
            # dataset each
            lp_trained_models = fine_tuning_res.models("model")

        if config.variation_category == "none":
            # Perform hyperparameter tuning
            eval_crossproduct = False
            (
                pretrained_models,
                ft_datasets,
                model_data_kwargs,
            ) = _create_lr_tuning_full_ft_setup(
                config,
                lp_trained_models,
                target_datasets,
            )
        else:
            eval_crossproduct = _eval_crossproduct(
                len(lp_trained_models),
                len(target_datasets),
            )
            pretrained_models, model_data_kwargs = _get_ft_model_infos(
                config,
                models,
                lp_trained_models,
                target_datasets,
                set_num_classes=True,
            )
            ft_datasets = target_datasets

        fine_tuning_res = multi_training(
            fine_tune_supervised,
            experiment_id,
            models=pretrained_models,
            datasets=ft_datasets,
            training_config=whole_model_ft_training_config,
            cross_product=eval_crossproduct,
            freeze_config=whole_model_ft_freeze_config,
            model_data_kwargs=model_data_kwargs,
        )

    training_metadata = {
        "--".join(model_key): metadata
        for model_key, metadata in training_res.metadata().items()
    }
    transfer_metadata = {
        "--".join(model_key): metadata
        for model_key, metadata in fine_tuning_res.metadata().items()
    }
    return TransformRelationshipResult(
        training_performance=training_res.metrics,
        training_metadata=training_metadata,
        transfer_performance=fine_tuning_res.metrics,
        transfer_metadata=transfer_metadata,
    )


def _get_ft_model_infos(
    config: ExperimentConfig,
    original_models: list[ModelInfo],
    pretrained_models: (
        Mapping[str, torch.nn.Module] | Mapping[ModelKey, torch.nn.Module]
    ),
    target_datasets: dict[str, MulticlassDataModule],
    set_num_classes: bool = False,
) -> tuple[list[ModelInfo], dict]:
    if _eval_crossproduct(len(pretrained_models), len(target_datasets)):
        # We're getting model infos for models that have not been fine-tuned
        # on the target datasets yet
        assert len(original_models) == len(pretrained_models)
        model_info_map = {
            model_info.name: model_info for model_info in original_models
        }
        combination_func = product
    else:
        # We're getting model infors for models that have already been
        # fine-tuned on the target datasets
        assert len(original_models) == 1
        model_info = original_models[0]
        model_info_map = {
            pt_model_name: model_info
            for pt_model_name in pretrained_models.keys()
        }
        combination_func = zip

    model_data_kwargs = {}
    model_infos = []
    for (model_name, model), (dataset_name, dataset) in combination_func(
        pretrained_models.items(), target_datasets.items()
    ):
        assert isinstance(model_name, str)
        model_info = model_info_map[model_name]
        model_config = model_info.config
        assert model_config is not None
        if set_num_classes:
            num_classes = dataset.num_classes
            model_config = deepcopy(model_config)
            model_config.num_classes = num_classes
        model_infos.append(
            ModelInfo(
                model=model,
                name=model_name,
                config=model_config,
            )
        )

        model_data_kwargs[(model_name, dataset_name)] = {
            "optimizer_config": _create_lr_optimizer_config(
                FULL_FT_LEARNING_RATES[model_config.type],
                config.fine_tuning_training.max_steps,
            ),
        }
    return model_infos, model_data_kwargs


def _eval_crossproduct(
    num_pretrained_models: int,
    num_target_datasets: int,
) -> bool:
    return num_pretrained_models != num_target_datasets


def _create_lr_tuning_full_ft_setup(
    config: ExperimentConfig,
    pretrained_models: (
        Mapping[str, torch.nn.Module] | Mapping[ModelKey, torch.nn.Module]
    ),
    target_datasets: dict[str, MulticlassDataModule],
) -> tuple[list[ModelInfo], dict[str, MulticlassDataModule], dict]:
    assert len(pretrained_models) == 1
    assert len(target_datasets) == 1
    pretrained_model = next(iter(pretrained_models.values()))
    model_config = config.model
    target_dataset = next(iter(target_datasets.values()))
    num_epochs = config.fine_tuning_training.max_steps

    model_data_kwargs = {}
    models = []
    datasets = {}
    for lr in HYPERPARAMETER_LR_GRID:
        lr_name = f"lr_{lr:.0e}"
        datasets[lr_name] = target_dataset
        model_config = deepcopy(model_config)
        model_config.num_classes = target_dataset.num_classes
        model_info = ModelInfo(
            model=pretrained_model,
            name=lr_name,
            config=model_config,
        )
        models.append(model_info)

        model_data_kwargs[(lr_name, lr_name)] = {
            "optimizer_config": _create_lr_optimizer_config(lr, num_epochs),
        }
    return models, datasets, model_data_kwargs


def _create_lr_optimizer_config(
    learning_rate: float, num_epochs: int
) -> OptimizerConfig:
    def optimizer_fn(model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(lr=learning_rate, params=model.parameters())

    def scheduler_fn(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return cast(
            torch.optim.lr_scheduler._LRScheduler,
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                last_epoch=num_epochs,
            ),
        )

    return OptimizerConfig(optimizer_fn, scheduler_fn)


def create_models(
    config: ModelConfig,
    variation_category: str,
    datasets: dict[str, MulticlassDataModule],
) -> list[ModelInfo]:
    if variation_category == "architecture":
        return create_architecture_variation_models(config, datasets.keys())
    elif variation_category == "class_relationship":
        return create_class_relationship_variation_models(
            config, datasets.items()
        )
    else:
        # n_samples
        return [
            ModelInfo(ds_name, create_model(config), deepcopy(config))
            for ds_name in datasets.keys()
        ]


def create_architecture_variation_models(
    model_config: ModelConfig,
    architectures: Iterable[str],
) -> list[ModelInfo]:
    models = []
    for architecture in architectures:
        architecture_config = deepcopy(model_config)
        architecture_config.type = architecture
        model = create_model(architecture_config)
        models.append(ModelInfo(architecture, model, architecture_config))
    return models


def create_class_relationship_variation_models(
    model_config: ModelConfig,
    datasets: Iterable[tuple[str, MulticlassDataModule]],
) -> list[ModelInfo]:
    models = []
    for class_relationship, dataset in datasets:
        class_relationship_config = deepcopy(model_config)
        class_relationship_config.num_classes = dataset.num_classes
        model = create_model(class_relationship_config)
        models.append(
            ModelInfo(class_relationship, model, class_relationship_config)
        )
    return models
