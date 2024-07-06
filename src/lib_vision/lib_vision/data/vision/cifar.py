import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Literal, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from lib_dl_base.io.dirs import get_dataset_dir

from ..lib.multiclass_datamodule import MulticlassDataModule
from ..loading import DataLoaderConfig, load
from ..wrappers.sample_wrapper import DatasetWrapper


CIFAR10_CLASS_LABELS = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

DEFAULT_TRAIN_TRANSFORMS = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
]
DEFAULT_TEST_TRANSFORMS = []

NORMALIZATION_TRANSFORMS = {
    "cifar10": transforms.Normalize(
        torch.tensor([0.4914, 0.4822, 0.4465]),
        torch.tensor([0.2023, 0.1994, 0.2010]),
    ),
    "cifar100": transforms.Normalize(
        torch.tensor([0.5071, 0.4865, 0.4409]),
        torch.tensor([0.2673, 0.2564, 0.2762]),
    ),
}

TRAIN_SPLIT = 0.9


# CifarType = Literal["cifar10", "cifar100"]
CifarType = str
AugmentationFunc = Callable[[Image.Image], Image.Image]


@dataclass
class CifarDataConfig:
    cifar_type: CifarType
    loader: DataLoaderConfig
    training_augmentations: list[AugmentationFunc] = field(
        default_factory=lambda: (list(DEFAULT_TRAIN_TRANSFORMS))
    )
    test_augmentations: list[AugmentationFunc] = field(
        default_factory=lambda: (list(DEFAULT_TEST_TRANSFORMS))
    )
    # training_label_transformations: Optional[Callable[[int], int]] = None
    train_sampler_constructor: Optional[Callable[[Dataset], Sampler]] = None
    test_sampler_constructor: Optional[Callable[[Dataset], Sampler]] = None
    to_tensor: bool = True
    normalize: bool = True


# Based on
# https://pytorch-lightning.readthedocs.io/en/latest/notebooks
# /lightning_examples/datamodules.html
class CifarData(MulticlassDataModule):
    def __init__(self, config: CifarDataConfig) -> None:
        assert config.cifar_type in ["cifar10", "cifar100"]
        super().__init__(
            classes=(
                CIFAR10_CLASS_LABELS if config.cifar_type == "cifar10" else 100
            )
        )

        self.data_dir = str(get_dataset_dir(config.cifar_type))
        self.transforms_train = config.training_augmentations
        self.transforms_test = config.test_augmentations
        self.loader_config = config.loader
        self.dataset_constructor = (
            CIFAR10 if config.cifar_type == "cifar10" else CIFAR100
        )
        self.train_sampler_constructor = config.train_sampler_constructor
        self.test_sampler_constructor = config.test_sampler_constructor
        self.norm_tensor_transforms = []
        if config.to_tensor:
            self.norm_tensor_transforms.append(transforms.ToTensor())
        if config.normalize:
            self.norm_tensor_transforms.append(
                NORMALIZATION_TRANSFORMS[config.cifar_type]
            )

    def prepare_data(self) -> None:
        # Download the dataset
        self.dataset_constructor(self.data_dir, train=True, download=True)
        self.dataset_constructor(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage == "train" or stage is None:
            # TODO: use the test transforms for the validation set
            cifar_full = self.dataset_constructor(
                self.data_dir,
                train=True,
                transform=transforms.Compose(
                    [
                        *self.transforms_train,
                        *self.norm_tensor_transforms,
                    ]
                ),
            )
            n_training_samples = len(cifar_full)
            split = int(math.floor(TRAIN_SPLIT * n_training_samples))
            train_split, val_split = random_split(
                cifar_full, [split, n_training_samples - split]
            )
            self.set_dataset("train", DatasetWrapper(train_split))
            self.set_dataset("val", DatasetWrapper(val_split))

        if stage == "test" or stage is None:
            self.set_dataset(
                "test",
                DatasetWrapper(
                    self.dataset_constructor(
                        self.data_dir,
                        train=False,
                        transform=transforms.Compose(
                            [
                                *self.transforms_test,
                                *self.norm_tensor_transforms,
                            ]
                        ),
                    )
                ),
            )

        if stage is not None and stage not in ["fit", "train", "test"]:
            raise ValueError(f"Invalid stage '{stage}'")

    def train_dataloader(self) -> DataLoader:
        dataset = self.get_dataset("train")
        return load(
            dataset,
            train=True,
            config=self._get_loader_config(
                dataset,
                self.train_sampler_constructor,
            ),
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self.get_dataset("val")
        return load(
            dataset,
            train=False,
            config=self._get_loader_config(
                dataset,
                self.test_sampler_constructor,
            ),
        )

    def test_dataloader(self) -> DataLoader:
        dataset = self.get_dataset("test")
        return load(
            dataset,
            train=False,
            config=self._get_loader_config(
                dataset,
                self.test_sampler_constructor,
            ),
        )

    def _get_loader_config(
        self,
        dataset: Dataset,
        sampler_constructor: Optional[Callable[[Dataset], Sampler]],
    ) -> DataLoaderConfig:
        if sampler_constructor:
            sampler = sampler_constructor(dataset)
            return DataLoaderConfig(
                **{
                    **asdict(self.loader_config),
                    "sampler": sampler,
                }
            )
        else:
            return self.loader_config
