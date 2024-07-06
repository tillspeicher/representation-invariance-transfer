from dataclasses import dataclass
from typing import Optional, Union

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

import lib_vision.data.loading as data_loading
from lib_vision.data.lib.dataset_accessor import DatasetStage
from lib_vision.data.lib.multiclass_datamodule import (
    LightningDataAccessor,
    MulticlassDataModule,
)

from .datasets import (
    BackgroundsDataset,
    BGConfig,
    Composite2DDataset,
    Composite2DDatasetConfig,
    FGConfig,
    ForegroundsDataset,
    ImageBackgroundsConfig,
    ImageForegroundsConfig,
    load_backgrounds,
    load_foregrounds,
)


FGSource = Union[FGConfig, MulticlassDataModule]
BGSource = Union[BGConfig, MulticlassDataModule]


@dataclass
class Transforms2DSamplingSeedsConfig:
    training: int
    validation: int
    test: int


@dataclass
class Transforms2DDataConfig:
    foregrounds: FGSource
    backgrounds: BGSource
    composition: Composite2DDatasetConfig
    n_training_samples: int
    n_val_samples: int
    n_test_samples: int
    sampling_seed: Union[int, Transforms2DSamplingSeedsConfig]
    transforms_sampling_seed: Optional[int]
    loader: data_loading.DataLoaderConfig


class Transforms2DData(MulticlassDataModule):
    def __init__(
        self,
        config: Transforms2DDataConfig,
    ) -> None:
        super().__init__(
            classes=(
                _get_num_foreground_classes(config.foregrounds)
                if config.composition.class_from_foreground
                else _get_num_background_classes(config.backgrounds)
            )
        )
        self.config = config

    def prepare_data(self):
        foregrounds = self.config.foregrounds
        backgrounds = self.config.backgrounds

        if isinstance(foregrounds, LightningDataAccessor):
            foregrounds.prepare_data()
        elif isinstance(foregrounds, ImageForegroundsConfig):
            load_foregrounds(
                img_size=foregrounds.img_size,
                use_single_object=foregrounds.use_single_object,
            )
        if isinstance(backgrounds, LightningDataAccessor):
            backgrounds.prepare_data()
        elif isinstance(backgrounds, ImageBackgroundsConfig):
            load_backgrounds(img_size=backgrounds.img_size)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is None or stage == "fit" or stage == "train":
            stage_type = "train"
            self.set_dataset(
                stage_type,
                _create_trans2d_dataset(
                    foregrounds_config=self.config.foregrounds,
                    backgrounds_config=self.config.backgrounds,
                    composite_config=self.config.composition,
                    n_samples=self.config.n_training_samples,
                    sampling_seed=_get_seed(
                        self.config.sampling_seed, stage_type
                    ),
                    transforms_sampling_seed=self.config.transforms_sampling_seed,
                    stage=stage_type,
                ),
            )
            stage_type = "val"
            self.set_dataset(
                stage_type,
                _create_trans2d_dataset(
                    foregrounds_config=self.config.foregrounds,
                    backgrounds_config=self.config.backgrounds,
                    composite_config=self.config.composition,
                    n_samples=self.config.n_val_samples,
                    sampling_seed=_get_seed(
                        self.config.sampling_seed, stage_type
                    ),
                    transforms_sampling_seed=self.config.transforms_sampling_seed,
                    stage=stage_type,
                ),
            )
        if stage is None or stage == "test":
            stage_type = "test"
            self.set_dataset(
                stage_type,
                _create_trans2d_dataset(
                    foregrounds_config=self.config.foregrounds,
                    backgrounds_config=self.config.backgrounds,
                    composite_config=self.config.composition,
                    n_samples=self.config.n_test_samples,
                    sampling_seed=_get_seed(
                        self.config.sampling_seed, stage_type
                    ),
                    transforms_sampling_seed=self.config.transforms_sampling_seed,
                    stage=stage_type,
                ),
            )

    def train_dataloader(self) -> DataLoader:
        return data_loading.load(
            self.get_dataset("train"),
            train=True,
            config=self.config.loader,
        )

    def val_dataloader(self) -> DataLoader:
        return data_loading.load(
            self.get_dataset("val"),
            train=False,
            config=self.config.loader,
        )

    def test_dataloader(self) -> DataLoader:
        return data_loading.load(
            self.get_dataset("test"),
            train=False,
            config=self.config.loader,
        )


def _create_trans2d_dataset(
    foregrounds_config: FGSource,
    backgrounds_config: BGSource,
    composite_config: Composite2DDatasetConfig,
    n_samples: int,
    sampling_seed: int,
    transforms_sampling_seed: Optional[int],
    stage: DatasetStage,
) -> Dataset:
    if isinstance(foregrounds_config, LightningDataAccessor):
        foregrounds_config.setup()
        return foregrounds_config.get_dataset(stage)
    else:
        foregrounds_data = ForegroundsDataset(
            foregrounds_config,
            n_samples=n_samples,
            sampling_seed=sampling_seed,
            transforms_sampling_seed=transforms_sampling_seed,
        )

    if isinstance(backgrounds_config, LightningDataAccessor):
        backgrounds_config.setup()
        backgrounds_data = backgrounds_config.get_dataset(stage)
    else:
        backgrounds_data = BackgroundsDataset(
            backgrounds_config,
            n_samples=n_samples,
            sampling_seed=sampling_seed + 593,
        )
    return Composite2DDataset(
        composite_config,
        foregrounds=foregrounds_data,
        backgrounds=backgrounds_data,
    )


def _get_seed(
    config: Union[int, Transforms2DSamplingSeedsConfig],
    data_type: DatasetStage,
) -> int:
    if data_type == "train":
        if isinstance(config, Transforms2DSamplingSeedsConfig):
            return config.training
        else:
            return config
    elif data_type == "val":
        if isinstance(config, Transforms2DSamplingSeedsConfig):
            return config.validation
        else:
            return config + 110
    elif data_type == "test":
        if isinstance(config, Transforms2DSamplingSeedsConfig):
            return config.test
        else:
            return config + 220
    else:
        raise ValueError()


def _get_num_foreground_classes(fg: FGSource) -> int:
    if isinstance(fg, MulticlassDataModule):
        return fg.num_classes
    elif isinstance(fg, ImageForegroundsConfig):
        return len(fg.fg_class_indices)
    else:
        return fg.n_classes


def _get_num_background_classes(bg: BGSource) -> int:
    if isinstance(bg, MulticlassDataModule):
        return bg.num_classes
    else:
        raise ValueError()
