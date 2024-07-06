from typing import Sized, TypeVar, Callable
from types import MethodType
import hashlib
import functools

import torch
import numpy as np
from torch.utils.data import Dataset

from ..lib.multiclass_datamodule import MulticlassDataModule
from ..lib.dataset_accessor import LightningDataAccessor, DatasetStage
from .data_sample import DataSample


D = TypeVar("D", bound=LightningDataAccessor)


def wrap_module(
    module: D,
    dataset_wrapper: Callable[[Dataset], Dataset],
    stages_to_wrap: list[DatasetStage] = ["train", "val", "test"],
) -> D:
    """Wraps a LightningDataModule to intercept the setup() method and
    wrap the inner datasets with wrappers."""
    prev_setup = module.setup

    @functools.wraps(module.setup)
    def wrapped_setup(self, stage: DatasetStage) -> None:
        prev_setup(stage)

        relevant_stages = LightningDataAccessor.get_relevant_stages(stage)
        for target_stage in relevant_stages:
            if target_stage in stages_to_wrap:
                self.set_dataset(
                    target_stage,
                    dataset_wrapper(self.get_dataset(target_stage)),
                )

    module.setup = MethodType(wrapped_setup, module)
    return module


def get_pseudorandom_idx(idx: int, n_items: int, salt: int) -> int:
    """Maps sample indices to item indices in a shuffled but deterministic way.
    I.e. for a given index, you will always get the same item index, but the
    mapping is different for different indices.

    Args:
        idx: The sample index
        n_items: The number of items in the dataset
        salt: A salt to make the mapping unique for each dataset

    Returns:
        The remapped item index
    """
    return (
        int.from_bytes(
            hashlib.sha256(bytes(idx + salt), usedforsecurity=False).digest(),
            byteorder="big",
            signed=False,
        )
        % n_items
    )


class ShuffledLablesDataset(Dataset):
    def __init__(self, dataset: Dataset, num_classes: int, seed: int):
        self.dataset = dataset
        self.num_classes = num_classes
        rng = np.random.default_rng(seed)
        self.salt = rng.integers(0, 2**24 - 1, dtype=np.uint32)

    def __getitem__(self, index: int) -> DataSample:
        x, _ = self.dataset[index]
        shuffled_y = get_pseudorandom_idx(index, self.num_classes, self.salt)
        return DataSample(
            input=x,
            target=torch.tensor(shuffled_y),
        )

    def __len__(self):
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        else:
            raise NotImplementedError("Length of dataset is not implemented")


def shuffle_labels(
    module: MulticlassDataModule, seed: int
) -> MulticlassDataModule:
    return wrap_module(
        module,
        lambda dataset: ShuffledLablesDataset(
            dataset, module.num_classes, seed
        ),
        stages_to_wrap=["train"],
    )
