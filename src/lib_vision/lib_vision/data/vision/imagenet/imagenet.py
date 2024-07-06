import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import datasets
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
from torchvision import transforms as transforms_lib
from torchvision.datasets import ImageNet

from lib_dl_base.io.dirs import get_dataset_dir

from ...lib.multiclass_datamodule import MulticlassDataModule
from ...loading import DataLoaderConfig, load
from ...wrappers.sample_wrapper import DatasetWrapper


@dataclass
class ImagenetDataConfig:
    """
    Args:
        data_dir: path to the imagenet dataset file
        meta_dir: path to meta.bin file
        num_imgs_per_val_class: how many images per class for the validation set
        image_size: final image size
        shuffle: If true shuffles the data every epoch
        drop_last: If true drops the last incomplete batch
    """

    loader: DataLoaderConfig
    data_dir: str = "<imagenet_storage_dir>"
    meta_dir: Optional[str] = None
    train_val_split: float = 0.961  # 3.9% validation data ~= 50 images per class
    image_size: int = 224
    # sampler_constructor: Optional[Callable[[Dataset], Sampler]] = None
    to_tensor: bool = True
    normalize: bool = True


def load_imagenet_hf():
    imagenet = datasets.load_dataset("imagenet2012")
    return imagenet


NUM_IMAGENET_SAMPLES = 1281167


# Based on https://github.com/Lightning-AI/lightning-bolts/blob/master
# /pl_bolts/datamodules/imagenet_datamodule.py
class ImagenetDataModule(MulticlassDataModule):
    """
    .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com
        /wp-content/uploads/2017/08/
            Sample-of-Images-from-the-ImageNet-Dataset-used-in-the
                -ILSVRC-Challenge.png
        :width: 400
        :alt: Imagenet
    Specs:
        - 1000 classes
        - Each image is (3 x varies x varies) (here we default to 3 x 224 x 224)
    Imagenet train, val and test dataloaders.
    The train set is the imagenet train.
    The val set is taken from the train set with `train_val_split`
    fraction of images.
    For example if `train_val_split=0.961` then there will be 2,000 images
    in the validation set.
    The test set is the official imagenet validation set.
    """

    name = "imagenet"

    def __init__(self, config: ImagenetDataConfig) -> None:
        # TODO: set the actual classses
        super().__init__(classes=1000)

        self.data_dir = Path(config.data_dir)
        self.meta_dir = Path(config.meta_dir) if config.meta_dir else None
        self.loader_config = config.loader

        self.image_size = config.image_size
        (
            self.num_train_samples,
            self.num_val_samples,
        ) = ImagenetDataModule._compute_num_train_val_samples(config.train_val_split)
        self.dims = (3, config.image_size, config.image_size)

        self.to_tensor = config.to_tensor
        self.normalize = config.normalize

    def prepare_data(self) -> None:
        """This method already assumes you have imagenet2012 downloaded.
        It validates the data using the meta.bin warning::
        Please download imagenet on your own first.
        """
        ImagenetDataModule._verify_splits(self.data_dir, "train")
        ImagenetDataModule._verify_splits(self.data_dir, "val")

        for split in ["train", "val"]:
            if not (self.data_dir / split / "meta.bin").exists():
                raise FileNotFoundError(
                    """
                no meta.bin present. Imagenet is no longer automatically
                downloaded by PyTorch.
                To get imagenet:
                1. download yourself from
                    http://www.image-net.org/challenges/LSVRC/2012/downloads
                2. download the devkit (ILSVRC2012_devkit_t12.tar.gz)
                3. generate the meta.bin file using the devkit
                4. copy the meta.bin file into both train and val split folders
                To generate the meta.bin do the following:
                from pl_bolts.datasets import UnlabeledImagenet
                path = '/path/to/folder/with/ILSVRC2012_devkit_t12.tar.gz/'
                UnlabeledImagenet.generate_meta_bins(path)
                """
                )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage == "train" or stage is None:
            dataset = ImageNet(
                root=str(self.data_dir),
                split="train",
            )
            train_dataset, val_dataset = random_split(
                dataset, [self.num_train_samples, self.num_val_samples]
            )
            train_dataset.transform = self._train_transforms()  # type: ignore
            val_dataset.transform = self._val_transforms()  # type: ignore

            self.set_dataset("train", DatasetWrapper(train_dataset))
            self.set_dataset("val", DatasetWrapper(val_dataset))

        if stage == "test" or stage is None:
            dataset = ImageNet(
                root=str(self.data_dir),
                split="val",
                transform=self._val_transforms(),
            )
            self.set_dataset("test", DatasetWrapper(dataset))

        if stage is not None and stage not in ["fit", "train", "test"]:
            raise ValueError(f"Invalid stage '{stage}'")

    def train_dataloader(self) -> DataLoader:
        """Uses the train split of imagenet2012 and puts away a portion
        of it for the validation split."""
        dataset = self.get_dataset("train")
        return load(
            dataset,
            train=True,
            config=self.loader_config,
        )

    def val_dataloader(self) -> DataLoader:
        """Uses the part of the train split of imagenet2012 that was
        not used for training"""
        dataset = self.get_dataset("val")
        return load(
            dataset,
            train=False,
            config=self.loader_config,
        )

    def test_dataloader(self) -> DataLoader:
        dataset = self.get_dataset("test")
        return load(
            dataset,
            train=False,
            config=self.loader_config,
        )

    def _train_transforms(self) -> Callable:
        """The standard imagenet transforms.
        .. code-block:: python
            transform_lib.Compose([
                transform_lib.RandomResizedCrop(self.image_size),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ToTensor(),
                transform_lib.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """
        return ImagenetDataModule._combine_transforms(
            [
                transforms_lib.RandomResizedCrop(self.image_size),
                transforms_lib.RandomHorizontalFlip(),
            ],
            to_tensor=self.to_tensor,
            normalize=self.normalize,
        )

    def _val_transforms(self) -> Callable:
        """The standard imagenet transforms for validation.
        .. code-block:: python
            transform_lib.Compose([
                transform_lib.Resize(self.image_size + 32),
                transform_lib.CenterCrop(self.image_size),
                transform_lib.ToTensor(),
                transform_lib.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """
        return ImagenetDataModule._combine_transforms(
            [
                transforms_lib.Resize(self.image_size + 32),
                transforms_lib.CenterCrop(self.image_size),
            ],
            to_tensor=self.to_tensor,
            normalize=self.normalize,
        )

    @staticmethod
    def _compute_num_train_val_samples(
        train_val_split: float,
    ) -> tuple[int, int]:
        split = int(math.floor(train_val_split * NUM_IMAGENET_SAMPLES))
        return split, NUM_IMAGENET_SAMPLES - split

    @staticmethod
    def _verify_splits(data_dir: Path, split: str) -> None:
        if not (data_dir / split).exists():
            raise FileNotFoundError(
                f"a {split} Imagenet split was not found in {data_dir},"
                f" make sure the folder contains a subfolder named {split}"
            )

    @staticmethod
    def _combine_transforms(
        transforms: list[Callable],
        to_tensor: bool,
        normalize: bool,
    ) -> Callable:
        """Combines the transforms into a single list of transforms.
        Args:
            transforms: the transforms to combine
            normalize: whether to add imagenet normalization
            to_tensor: whether to add to tensor
        """
        if to_tensor:
            transforms.append(transforms_lib.ToTensor())
        if normalize:
            transforms.append(
                transforms_lib.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                )
            )
        return transforms_lib.Compose(transforms)
