from collections import defaultdict
from typing import Iterator, Union

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class ReducedSampler(Sampler):
    def __init__(self, seed: int) -> None:
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.indices: list[int]

    def __iter__(self) -> Iterator[int]:
        permutation = torch.randperm(
            len(self.indices), generator=self.generator
        ).tolist()
        yield from (self.indices[i] for i in permutation)

    def __len__(self) -> int:
        return len(self.indices)


class ClassDataReducedSampler(ReducedSampler):
    """Samples indices from a reduced number of classes"""

    def __init__(
        self,
        data_source: Dataset,
        seed: int,
        retain_classes: Union[float, int, list[int]] = 1.0,
        retain_samples_per_class: Union[float, int] = 1.0,
    ) -> None:
        super().__init__(seed)

        class_indices = _get_class_indices(data_source)
        if isinstance(retain_classes, list):
            dataset_classes = list(class_indices.keys())
            assert all(c_idx in dataset_classes for c_idx in retain_classes), (
                f"Not all indices in {retain_classes} in the dataset "
                f"class indices {dataset_classes}"
            )
            self.retained_classes = retain_classes
        else:
            self.retained_classes = _subsample(
                list(class_indices.keys()),
                retain_classes,
                generator=self.generator,
            )

        # Collect the indices of all samples that should be retained,
        # i.e. the indices of samples belonging to retained classes
        self.indices = [
            index
            for i_class in self.retained_classes
            for index in _subsample(
                class_indices[i_class],
                retain_samples_per_class,
                generator=self.generator,
            )
        ]


def _get_class_indices(dataset: Dataset) -> dict[int, list[int]]:
    # We do this to disable class remapping in the RWConsistentSampler
    # from the representation invariance project, slightly hacky
    dataset.use_class_remapping = False  # type: ignore
    class_indices = defaultdict(list)
    for i in range(len(dataset)):  # type: ignore
        sample = dataset[i]
        class_indices[sample.target].append(i)
    dataset.use_class_remapping = True  # type: ignore
    return class_indices


def _subsample(
    items: list[int],
    retain_amount: Union[float, int],
    generator: torch.Generator,
) -> list[int]:
    perm = torch.randperm(len(items), generator=generator).tolist()
    if isinstance(retain_amount, float):
        cutoff = int(round(retain_amount * len(items)))
    else:
        cutoff = retain_amount
    assert cutoff <= len(
        items
    ), f"Cannot retain {cutoff} items from {len(items)}"
    return [items[i] for i in perm[:cutoff]]
