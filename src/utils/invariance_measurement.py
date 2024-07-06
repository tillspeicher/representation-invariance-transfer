from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import transforms_2d as t2d
from lib_dl_base.defs.task_id import TaskID
from lib_vision.data.lib.multiclass_datamodule import MulticlassDataModule
from lib_vision.models.architectures import ModelConfig
from lib_vision.models.metrics.representations.invariance import (
    InvarianceMeasurementConfig,
    multi_measure_invariance,
)
from utils.data import (  # create_rw_dataset,; CifarDataConfig,; DataLoaderConfig,
    create_t2d_like_rw_dataset,
)


@dataclass
class STDataConfig(t2d.Transforms2DConfig):
    # Used to determine the structure of the dataset, i.e. which transforms
    # and classes to use
    config_seed: int
    n_classes: int
    foreground_types: list[str]
    background_types: list[str]
    transforms: Optional[list[str]] = None
    # including_random: bool = False


@dataclass
class SampledDataConfig:
    objects: list[int]


FOREGROUND_TYPES = {
    "op": "objects-primary",
    "os": "objects-secondary",
    "rp": "random-primary",
    "rs": "random-secondary",
}
BACKGROUND_TYPES = {
    "img": "images",
    "rand": "random",
}


def create_single_transform_datasets(
    config: STDataConfig,
    transforms_sampling_seed: Optional[int] = None,
    include_none: bool = False,
) -> dict[str, dict[str, MulticlassDataModule]]:
    """Create datasets with a single transform"""
    assert len(config.foreground_types) == len(config.background_types)

    if transforms_sampling_seed is not None:
        config = deepcopy(config)
        config.transforms_sampling_seed = transforms_sampling_seed
    transforms = (
        config.transforms
        if config.transforms is not None
        else list(t2d.TRANSFORMS)
    )
    if include_none:
        transforms.append("none")

    datasets = {}
    for foreground_type, background_type in zip(
        config.foreground_types,
        config.background_types,
    ):
        foregrounds = _create_foregrounds(
            foreground_type,
            config_seed=config.config_seed,
            n_classes=config.n_classes,
        )
        backgrounds = _create_backgrounds(
            background_type,
            config_seed=config.config_seed,
        )
        data_type_datasets: dict[str, MulticlassDataModule] = {
            transform: t2d.create_t2d_dataset(
                transforms=[transform],
                config=config,
                foregrounds=foregrounds,
                backgrounds=backgrounds,
            )
            for transform in transforms
        }
        datasets[f"{foreground_type}-{background_type}"] = data_type_datasets
    return datasets


def _create_foregrounds(
    foreground_type: str,
    config_seed: int,
    n_classes: int,
) -> t2d.FGSource:
    if foreground_type in ["op", "os"]:
        rng = np.random.default_rng(config_seed)
        prim_foregrounds, sec_foregrounds = _get_classes(rng, n_classes)
        if foreground_type == "op":
            foregrounds = prim_foregrounds
        else:
            foregrounds = sec_foregrounds
        return t2d.ImageForegroundsConfig(
            fg_class_indices=foregrounds,
            img_size=-1,  # Will be overwritten by the create function
            transforms=[],  # Will be overwritten by the create function
        )
    elif foreground_type in ["rp", "rs"]:
        if foreground_type == "rp":
            rand_config_seed = config_seed + 593
        else:
            rand_config_seed = config_seed + 245
        return t2d.RandomForegroundsConfig(
            n_classes=n_classes,
            img_size=-1,  # Will be overwritten by the create function
            transforms=[],  # Will be overwritten by the create function
            seed=rand_config_seed,
        )
    else:
        raise ValueError(f"Unknown foreground type {foreground_type}")


def _create_backgrounds(
    background_type: str,
    config_seed: int,
) -> t2d.BGSource:
    if background_type == "img":
        return t2d.ImageBackgroundsConfig(
            img_size=-1,  # Will be overwritten by the create function
        )
    elif background_type == "rand":
        return t2d.RandomBackgroundsConfig(
            seed=config_seed + 324,
            img_size=-1,  # Will be overwritten by the create function
        )
    else:
        raise ValueError(f"Unknown background type {background_type}")


def _get_classes(
    rng: np.random.Generator,
    n_classes: int,
) -> tuple[list[int], list[int]]:
    class_candidates = t2d.OBJECTS
    assert len(class_candidates) >= 2 * n_classes
    # Also choose an alternative set of OOD classes
    objects = list(rng.choice(class_candidates, 2 * n_classes, replace=False))
    return objects[:n_classes], objects[n_classes:]


def create_rw_datasets(
    data_type: str,
    sampling_seed: int,
    transforms_sampling_seed: int,
    normalize: bool = True,
) -> dict[str, MulticlassDataModule]:
    transforms = t2d.TRANSFORMS
    if data_type == "cifar10":
        classes = list(range(10))
    elif data_type == "cifar100":
        classes = list(range(100))
    else:
        raise ValueError(f"Unknown data type {data_type}")
    return {
        transform: create_t2d_like_rw_dataset(
            data_type=data_type,
            transforms=[transform],
            sampling_seed=sampling_seed,
            transforms_sampling_seed=transforms_sampling_seed,
            classes=classes,
            normalize=normalize,
        )
        for transform in transforms
    }


def measure_single_transform_invariance(
    task_id: TaskID,
    config: InvarianceMeasurementConfig,
    models: dict[str, torch.nn.Module],
    model_config: Union[ModelConfig, dict[str, ModelConfig]],
    data_config: STDataConfig,
    transforms_sampling_seeds: tuple[int, int],
    use_rw_data: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Measures invariance to single transformations"""
    # Create two versions of the data that only differ in their transformations
    # This allows us to compare how much the model's representations change
    # based on the transformation applied to the objects, i.e. how invariant
    # they are to the transformation.

    datasets_1 = {}
    datasets_2 = {}
    if use_rw_data in ["no", "both"]:
        datasets_1 |= create_single_transform_datasets(
            data_config, transforms_sampling_seeds[0]
        )
        datasets_2 |= create_single_transform_datasets(
            data_config, transforms_sampling_seeds[1]
        )
    if use_rw_data in ["only", "both"]:
        for data_type in ["cifar10", "cifar100"]:
            datasets_1[data_type] = create_rw_datasets(
                data_type,
                data_config.sampling_seed,
                transforms_sampling_seeds[0],
            )
            datasets_2[data_type] = create_rw_datasets(
                data_type,
                data_config.sampling_seed,
                transforms_sampling_seeds[1],
            )

    invariances: dict[str, dict[str, pd.DataFrame]] = {}
    for data_type, data_type_dataset_1 in datasets_1.items():
        print("Measuring invariance for data type", data_type)
        data_type_dataset_2 = datasets_2[data_type]
        data_type_invariances = multi_measure_invariance(
            task_id=task_id.add_prefix(data_type),
            config=config,
            models=models,
            model_configs=model_config,
            datasets_1=data_type_dataset_1,
            datasets_2=data_type_dataset_2,
        )
        invariances[data_type] = data_type_invariances
    return invariances
