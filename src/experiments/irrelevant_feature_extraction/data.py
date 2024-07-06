from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Optional, cast

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import transforms_2d as t2d
from lib_vision.data.lib.multiclass_datamodule import MulticlassDataModule
from lib_vision.data.loading import DataLoaderConfig
from lib_vision.data.vision.cifar import CifarData, CifarDataConfig
from transforms_2d.transforms import (
    ImageWrapper,
    scale_object,
    translate_object,
    with_image_wrapper,
)


@dataclass
class IFEDataConfig(t2d.Transforms2DConfig):
    config_seed: int
    # Used to draw samples from the dataset
    n_classes: int


@dataclass
class IFEDatasets:
    datasets: dict[str, L.LightningDataModule]
    objects: list[int]


CORRELATION_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0]
AVAILABILITY_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8]

OBJECTS_ONLY_KEY = "objects_only"
MIXED_OBJECTS_KEY = "objects_mixed"
MIXED_CIFAR_COR_KEY = "cifar_mixed_cor"
MIXED_CIFAR_AVAILABILITY_KEY = "cifar_mixed_av"


def create_irrelevant_features_dataset(
    config: IFEDataConfig,
    transform_names: list[str],
    normalize: bool = True,
) -> IFEDatasets:
    loader_config = DataLoaderConfig(
        batch_size=config.batch_size,
    )

    rng = np.random.default_rng(config.config_seed)
    # Subsample foreground objects
    n_classes = config.n_classes
    assert n_classes == 10
    objects = [
        int(obj) for obj in rng.choice(t2d.OBJECTS, n_classes, replace=False)
    ]
    transforms = _create_transforms(transform_names)

    # A dataset with CIFAR images overlayed with random objects that uses
    # the CIFAR labels as targets
    mixed_cifar_labels_availability_datasets = {
        f"{MIXED_CIFAR_AVAILABILITY_KEY}{fg_av}": _create_mixed_dataset(
            config,
            transforms,
            objects,
            loader_config,
            # The CIFAR images are the backgrounds
            class_from_foreground=False,
            fg_availability=fg_av,
            normalize=normalize,
        )
        for fg_av in AVAILABILITY_VALUES
    }
    mixed_cifar_labels_correlation_datasets = {
        f"{MIXED_CIFAR_COR_KEY}{fg_bg_correlation}": _create_mixed_dataset(
            config,
            transforms,
            objects,
            loader_config,
            # The CIFAR images are the backgrounds
            class_from_foreground=False,
            fg_bg_correlation=fg_bg_correlation,
            normalize=normalize,
        )
        for fg_bg_correlation in CORRELATION_VALUES
    }
    mixed_cifar_labels_datasets = {
        **mixed_cifar_labels_availability_datasets,
        **mixed_cifar_labels_correlation_datasets,
    }
    # A dataset with CIFAR images overlayed with random objects that uses
    # the random object types as targets
    mixed_object_labels_dataset = _create_mixed_dataset(
        config,
        transforms,
        objects,
        loader_config,
        # The objects images are the foregrounds
        class_from_foreground=True,
        normalize=normalize,
    )

    objects_only_dataset = t2d.create_t2d_dataset(
        config,
        transforms,
        objects,
        backgrounds=t2d.UniformBackgroundsConfig(
            [0, 0, 0], img_size=config.img_size
        ),
        normalize=normalize,
    )

    datasets: dict[str, L.LightningDataModule] = {
        # "mixed_cifar_labels": mixed_cifar_labels_dataset,
        **mixed_cifar_labels_datasets,
        MIXED_OBJECTS_KEY: mixed_object_labels_dataset,
        OBJECTS_ONLY_KEY: objects_only_dataset,
    }
    return IFEDatasets(
        datasets=datasets,
        objects=objects,
    )


def _scale_down(img: ImageWrapper) -> ImageWrapper:
    return scale_object(img, 0.55)


def _move_to_upper_right(img: ImageWrapper) -> ImageWrapper:
    return translate_object(img, 21, 2)


SCALE_DOWN_TRANSFORM = transforms.Lambda(_scale_down)
UPPER_RIGHT_CORNER = transforms.Lambda(_move_to_upper_right)


def _create_transforms(transform_names: list[str]) -> list[torch.nn.Module]:
    transforms = [SCALE_DOWN_TRANSFORM, UPPER_RIGHT_CORNER]
    for transform_name in transform_names:
        if transform_name != "none":
            transform = t2d.get_transform(transform_name)
            transforms.append(transform)
    return cast(list[torch.nn.Module], transforms)


def _create_mixed_dataset(
    config: IFEDataConfig,
    transforms: list[torch.nn.Module],
    fg_objects: list[int],
    loader_config: DataLoaderConfig,
    class_from_foreground: bool,
    fg_bg_correlation: float = 0.0,
    fg_availability: float = 1.0,
    normalize: bool = True,
) -> MulticlassDataModule:
    # Don't normalize and convert to tensor, the composite part of
    cifar_dataset = CifarData(
        CifarDataConfig(
            cifar_type="cifar10",
            loader=loader_config,
            to_tensor=False,
            normalize=False,
        )
    )
    assert fg_bg_correlation == 0 or (
        not class_from_foreground and len(fg_objects) == 10
    )
    return t2d.create_t2d_dataset(
        config,
        transforms,
        fg_objects,
        backgrounds=cifar_dataset,
        class_from_foreground=class_from_foreground,
        fg_bg_correlation=fg_bg_correlation,
        fg_availability=fg_availability,
        normalize=normalize,
    )


def create_invariance_measurement_datasets(
    config: IFEDataConfig,
    objects: list[int],
    normalize: bool = True,
) -> dict[str, list[MulticlassDataModule]]:
    cifar_datasets = _create_alternative_sampling_cifar_datasets(
        config.sampling_seed
    )
    t2d_variation_configs = []
    for seed_offset in range(2):
        seed_config = deepcopy(config)
        seed_config.sampling_seed += seed_offset + 10
        t2d_variation_configs.append(seed_config)
    transforms = _create_transforms([])
    return {
        "cifar": [
            t2d.create_t2d_dataset(
                t2d_variation_configs[0],
                transforms,
                objects,
                backgrounds=cifar_dataset,
                normalize=normalize,
            )
            for cifar_dataset in cifar_datasets
        ],
        "objects": [
            t2d.create_t2d_dataset(
                variation_config,
                transforms,
                objects,
                backgrounds=cifar_datasets[0],
                normalize=normalize,
            )
            for variation_config in t2d_variation_configs
        ],
    }


def _create_alternative_sampling_cifar_datasets(
    sampling_seed: int,
) -> list[MulticlassDataModule]:
    cifar_datasets = []
    for seed_offset in range(2):
        cifar_config = CifarDataConfig(
            cifar_type="cifar10",
            loader=DataLoaderConfig(
                batch_size=512,
            ),
            to_tensor=False,
            normalize=False,
        )
        cifar_data = RandomizedSamplingWrapper(
            CifarData(cifar_config),
            sampling_seed=sampling_seed + seed_offset,
        )
        cifar_datasets.append(cifar_data)
    return cifar_datasets


class RandomizedSamplingWrapper(MulticlassDataModule):
    def __init__(
        self,
        wrapped_ds: MulticlassDataModule,
        sampling_seed: int,
    ):
        super().__init__(wrapped_ds.num_classes)
        self.wrapped_ds = wrapped_ds
        self.sampling_seed = sampling_seed

    def setup(self, stage: Optional[str] = None):
        # TODO: somewhat duplicated with the consistent sampling wrapper
        # in utils.data.py
        self.wrapped_ds.setup(stage=stage)

        if stage is None:
            stages_to_wrap = ["train", "val", "test"]
        elif stage == "fit" or stage == "train":
            stages_to_wrap = ["train", "val"]
        elif stage == "test":
            stages_to_wrap = ["test"]
        else:
            raise ValueError(f"Unknown stage {stage}")
        for stage_to_wrap in stages_to_wrap:
            self.wrapped_ds.set_dataset(
                stage_to_wrap,
                RandomizedSamplingDataset(
                    wrapped_ds=self.wrapped_ds.get_dataset(stage_to_wrap),
                    sampling_seed=self.sampling_seed,
                ),
            )

    def get_dataset(self, stage: str) -> Dataset:
        return self.wrapped_ds.get_dataset(stage)

    def train_dataloader(self) -> DataLoader:
        return self.wrapped_ds.train_dataloader()

    def val_dataloader(self) -> DataLoader:
        return self.wrapped_ds.val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return self.wrapped_ds.test_dataloader()


class RandomizedSamplingDataset(Dataset):
    def __init__(
        self,
        wrapped_ds: Dataset,
        sampling_seed: int,
    ):
        super().__init__()
        self.wrapped_ds = wrapped_ds
        rng = np.random.default_rng(sampling_seed)
        self.sampling_order = rng.permutation(len(self.wrapped_ds))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        remapped_idx = self.sampling_order[index]
        return self.wrapped_ds[remapped_idx]

    def __len__(self) -> int:
        return len(self.wrapped_ds)
