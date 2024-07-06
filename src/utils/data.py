from copy import deepcopy
from typing import Optional, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import transforms_2d as t2d
from lib_vision.data.lib.multiclass_datamodule import MulticlassDataModule
from lib_vision.data.loading import DataLoaderConfig
from lib_vision.data.utils.samplers import ClassDataReducedSampler
from lib_vision.data.vision.cifar import CifarData, CifarDataConfig
from lib_vision.data.vision.imagenet.imagenet import (
    ImagenetDataConfig,
    ImagenetDataModule,
)
from transforms_2d.datasets import use_local_rng


RWDataConfig = Union[CifarDataConfig, ImagenetDataConfig]


def create_rw_dataset(
    config: RWDataConfig,
) -> MulticlassDataModule:
    if isinstance(config, CifarDataConfig):
        return CifarData(config)
    elif isinstance(config, ImagenetDataConfig):
        return ImagenetDataModule(config)
    else:
        raise NotImplementedError()


def get_dataset_name(config: RWDataConfig) -> str:
    if isinstance(config, CifarDataConfig):
        return config.cifar_type
    elif isinstance(config, ImagenetDataConfig):
        return "imagenet"
    else:
        raise ValueError(f"Unknown dataset type {config.dataset}")


def create_t2d_like_rw_dataset(
    data_type: str,
    transforms: list[str],
    sampling_seed: int,
    transforms_sampling_seed: int,
    classes: list[int],
    subsample: Union[float, int] = 1.0,
    normalize: bool = True,
) -> MulticlassDataModule:
    def train_sampler_constructor(dataset):
        return ClassDataReducedSampler(
            dataset,
            retain_classes=classes,
            retain_samples_per_class=subsample,
            seed=sampling_seed,
        )

    def test_sampler_constructor(dataset):
        return ClassDataReducedSampler(
            dataset,
            retain_classes=classes,
            retain_samples_per_class=1.0,
            seed=sampling_seed,
        )

    cifar_conf = CifarDataConfig(
        cifar_type=data_type,
        loader=DataLoaderConfig(
            batch_size=512,
        ),
        train_sampler_constructor=train_sampler_constructor,
        test_sampler_constructor=test_sampler_constructor,
        normalize=normalize,
    )
    return RWConsistentSamplingWrapper(
        create_rw_dataset(
            _set_augmentations(
                cifar_conf,
                transforms,
            )
        ),
        transforms_sampling_seed=transforms_sampling_seed,
        retained_classes=classes,
    )


def _set_augmentations(
    conf: CifarDataConfig,
    transforms: list[str],
) -> CifarDataConfig:
    augmentations = [t2d.get_unwrapped_transform(t) for t in transforms] + [
        rgba_to_rgb
    ]
    aug_conf = deepcopy(conf)
    aug_conf.training_augmentations = augmentations
    aug_conf.test_augmentations = augmentations
    return aug_conf


def rgba_to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


class RWConsistentSamplingWrapper(MulticlassDataModule):
    def __init__(
        self,
        wrapped_ds: MulticlassDataModule,
        transforms_sampling_seed: int,
        # Optionally, reduce the number of classes of the dataset, so
        # remap the class indices
        retained_classes: Optional[list[int]] = None,
    ):
        super().__init__(
            wrapped_ds.num_classes
            if retained_classes is None
            else len(retained_classes)
        )
        self.wrapped_ds = wrapped_ds
        self.transforms_sampling_seed = transforms_sampling_seed
        if retained_classes is not None:
            self.class_remapping = {
                c_idx: i for i, c_idx in enumerate(retained_classes)
            }
        else:
            self.class_remapping = None

    def setup(self, stage: Optional[str] = None):
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
                RWConsistentSamplingDataset(
                    wrapped_ds=self.wrapped_ds.get_dataset(stage_to_wrap),
                    transforms_sampling_seed=self.transforms_sampling_seed,
                    class_remapping=self.class_remapping,
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


class RWConsistentSamplingDataset(Dataset):
    def __init__(
        self,
        wrapped_ds: Dataset,
        transforms_sampling_seed: int,
        class_remapping: Optional[dict[int, int]],
    ):
        super().__init__()
        self.wrapped_ds = wrapped_ds
        # Set by the sampler. Slightly hacky
        self.use_class_remapping = True
        self.class_remapping = class_remapping
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(transforms_sampling_seed)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with use_local_rng(self.torch_rng):
            sample = self.wrapped_ds[index]
            if self.class_remapping is not None and self.use_class_remapping:
                remapped_target = self.class_remapping[sample.target]
                sample.target = remapped_target
            return sample

    def __len__(self) -> int:
        return len(self.wrapped_ds)
