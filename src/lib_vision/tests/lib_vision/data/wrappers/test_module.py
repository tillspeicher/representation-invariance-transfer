from typing import DefaultDict

import torch
from torch.utils.data import DataLoader, Dataset

from lib_vision.data.lib.multiclass_datamodule import MulticlassDataModule
from lib_vision.data.loading import DataLoaderConfig, load
from lib_vision.data.wrappers.data_sample import DataSample
from lib_vision.data.wrappers.module import (
    get_pseudorandom_idx,
    shuffle_labels,
    wrap_module,
)


NUM_SAMPLES = 32


class TestDataset(Dataset[DataSample]):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __getitem__(self, idx):
        return DataSample(
            input=torch.tensor(idx + 1),
            target=torch.tensor((idx + 2) % self.num_classes),
        )

    def __len__(self):
        return NUM_SAMPLES


class TestModule(MulticlassDataModule):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(classes=num_classes)

    def setup(self, stage: str):
        if stage == "fit" or stage == "train":
            stages = ["train", "val"]
        elif stage == "test":
            stages = ["test"]
        else:
            raise ValueError(f"Unknown stage {stage}")
        for stage_to_set in stages:
            self.set_dataset(stage_to_set, TestDataset(self.num_classes))

    def train_dataloader(self):
        return self._get_loader("train")

    def val_dataloader(self):
        return self._get_loader("val")

    def test_dataloader(self):
        return self._get_loader("test")

    def _get_loader(self, stage: str) -> DataLoader:
        dataset = self.get_dataset(stage)
        return load(
            dataset,
            train=False,
            config=DataLoaderConfig(batch_size=NUM_SAMPLES, num_workers=0),
        )


def test_wrap_module():
    class ConstantDataset(Dataset[DataSample]):
        def __getitem__(self, idx):
            return DataSample(
                input=torch.tensor(0),
                target=torch.tensor(0),
            )

        def __len__(self):
            return NUM_SAMPLES

    data = TestModule(10)
    data.setup("train")
    loader = data.train_dataloader()
    sample = next(iter(loader))
    torch.testing.assert_close(sample.input, torch.arange(1, NUM_SAMPLES + 1))
    torch.testing.assert_close(
        sample.target, torch.arange(2, NUM_SAMPLES + 2) % 10
    )

    wrapped_data = wrap_module(data, lambda _: ConstantDataset())
    wrapped_data.setup("test")
    wrapped_loader = wrapped_data.test_dataloader()
    wrapped_sample = next(iter(wrapped_loader))
    torch.testing.assert_close(
        wrapped_sample.input, torch.zeros(NUM_SAMPLES, dtype=torch.int64)
    )
    torch.testing.assert_close(
        wrapped_sample.target, torch.zeros(NUM_SAMPLES, dtype=torch.int64)
    )


def test_get_pseudorandom_idx():
    idx1 = get_pseudorandom_idx(2, 20, 45932)
    assert idx1 == get_pseudorandom_idx(2, 20, 45932)
    assert idx1 != get_pseudorandom_idx(3, 20, 45933)


def test_shuffle_labels():
    data1 = TestModule(10)
    data1.setup("train")
    loader = data1.train_dataloader()
    sample = next(iter(loader))

    data2 = TestModule(10)
    wrapped_data = shuffle_labels(data2, seed=2728)
    wrapped_data.setup("train")
    wrapped_loader = wrapped_data.train_dataloader()
    wrapped_sample = next(iter(wrapped_loader))

    assert torch.all(sample.input == wrapped_sample.input)
    assert not torch.all(sample.target == wrapped_sample.target)

    # Check that the labels are shuffled on a per-instance bases, not
    # just remapped consistently.
    orig_label_images = {i: [] for i in range(10)}
    shuffled_label_images = {i: [] for i in range(10)}
    for i in range(NUM_SAMPLES):
        orig_label_images[sample.target[i].item()].append(i)
        shuffled_label_images[wrapped_sample.target[i].item()].append(i)
    orig_partition = set(tuple(v) for v in orig_label_images.values())
    shuffled_partition = set(
        tuple(sorted(v)) for v in shuffled_label_images.values()
    )
    assert orig_partition != shuffled_partition


def test_shuffle_labels_repeat():
    seed = 9432
    data_1 = TestModule(15)
    wrapped_data_1 = shuffle_labels(data_1, seed=seed)
    wrapped_data_1.setup("train")
    loader_1 = wrapped_data_1.train_dataloader()
    sample_1 = next(iter(loader_1))

    data_2 = TestModule(15)
    wrapped_data_2 = shuffle_labels(data_2, seed=seed)
    wrapped_data_2.setup("train")
    loader_2 = wrapped_data_2.train_dataloader()
    sample_2 = next(iter(loader_2))

    assert torch.all(sample_1.input == sample_2.input)
    assert torch.all(sample_1.target == sample_2.target)


def test_shuffling_repeats_on_resets():
    seed = 5773
    data = TestModule(10)
    wrapped_data = shuffle_labels(data, seed=seed)
    torch.manual_seed(593)
    wrapped_data.setup("train")
    ds_1 = wrapped_data.get_dataset("train")
    labels_1 = [ds_1[i].target for i in range(len(ds_1))]
    torch.manual_seed(230)
    wrapped_data.setup("train")
    ds_2 = wrapped_data.get_dataset("train")
    # Check that things don't depend on the iteration order
    labels_2 = [ds_2[i].target for i in reversed(range(len(ds_2)))]
    assert labels_1 == labels_2[::-1]


def test_shuffling_repeats_on_iterations():
    seed = 59802
    torch.manual_seed(593)
    data = TestModule(10)
    wrapped_data = shuffle_labels(data, seed=seed)
    wrapped_data.setup("train")
    loader = wrapped_data.train_dataloader()
    label_counts_1 = DefaultDict(int)
    for batch in loader:
        for label in batch.target:
            label_counts_1[label.item()] += 1
    # Iterate a second time and check whether the counts match
    label_counts_2 = DefaultDict(int)
    for batch in loader:
        for label in batch.target:
            label_counts_2[label.item()] += 1
    assert label_counts_1 == label_counts_2


def test_shuffle_only_train():
    seed = 593
    orig_data = TestModule(15)
    orig_data.setup("train")
    orig_data.setup("test")
    train_loader_1 = orig_data.train_dataloader()
    train_sample_1 = next(iter(train_loader_1))
    val_loader_1 = orig_data.val_dataloader()
    val_sample_1 = next(iter(val_loader_1))
    test_loader_1 = orig_data.test_dataloader()
    test_sample_1 = next(iter(test_loader_1))

    data_2 = TestModule(15)
    wrapped_data = shuffle_labels(data_2, seed=seed)
    wrapped_data.setup("train")
    wrapped_data.setup("test")
    train_loader_2 = wrapped_data.train_dataloader()
    train_sample_2 = next(iter(train_loader_2))
    val_loader_2 = wrapped_data.val_dataloader()
    val_sample_2 = next(iter(val_loader_2))
    test_loader_2 = wrapped_data.test_dataloader()
    test_sample_2 = next(iter(test_loader_2))

    assert torch.all(train_sample_1.input == train_sample_2.input)
    assert not torch.all(train_sample_1.target == train_sample_2.target)
    assert torch.all(val_sample_1.input == val_sample_2.input)
    assert torch.all(val_sample_1.target == val_sample_2.target)
    assert torch.all(test_sample_1.input == test_sample_2.input)
    assert torch.all(test_sample_1.target == test_sample_2.target)
