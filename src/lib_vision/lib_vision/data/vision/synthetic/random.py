from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Union

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch

from ... import loading as data_loading
from ...lib.dataset_accessor import (
    LightningDataAccessor,
)
from ...wrappers.data_sample import DataSample


@dataclass
class RandomDataConfig:
    img_size: int
    n_training_samples: int
    n_val_samples: int
    n_test_samples: int
    sampling_seed: int
    loader: data_loading.DataLoaderConfig


class RandomDataModule(LightningDataAccessor):
    def __init__(
        self,
        config: RandomDataConfig,
    ) -> None:
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is None or stage == "fit" or stage == "train":
            stage_type = "train"
            self.set_dataset(
                stage_type,
                RandomDataset(
                    self.config.img_size,
                    n_samples=self.config.n_training_samples,
                    sampling_seed=self.config.sampling_seed,
                ),
            )
            stage_type = "val"
            self.set_dataset(
                stage_type,
                RandomDataset(
                    self.config.img_size,
                    n_samples=self.config.n_val_samples,
                    sampling_seed=self.config.sampling_seed + 1,
                ),
            )
        if stage is None or stage == "test":
            stage_type = "test"
            self.set_dataset(
                stage_type,
                RandomDataset(
                    self.config.img_size,
                    n_samples=self.config.n_test_samples,
                    sampling_seed=self.config.sampling_seed + 2,
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


class RandomDataset(Dataset[DataSample[torch.Tensor]]):
    def __init__(
        self,
        img_size: int,
        n_samples: int,
        sampling_seed: int,
    ) -> None:
        self.img_size = img_size
        self.n_samples = n_samples
        self.sampling_seed = sampling_seed

        self.rng = torch.Generator()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> DataSample[torch.Tensor]:
        seed = self.sampling_seed + index
        self.rng.manual_seed(seed)
        return DataSample(
            input=torch.rand(
                3,
                self.img_size,
                self.img_size,
                generator=self.rng,
                dtype=torch.float32,
            )
        )
