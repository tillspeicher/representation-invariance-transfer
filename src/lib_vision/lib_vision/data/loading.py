import multiprocessing
from dataclasses import dataclass
from typing import Optional

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .wrappers.data_sample import DataSample


n_gpus = torch.cuda.device_count()
n_cpus = multiprocessing.cpu_count()
# Distribute the number of loader cores evenly between the GPUs
# If there are no GPUs leave on CPU for the processing work
n_available_cpus = int(n_cpus / n_gpus) if n_gpus > 0 else n_cpus - 1
MAX_CPU_WORKERS = 12


@dataclass
class DataLoaderConfig:
    batch_size: int  # = 128
    num_workers: int = min(n_available_cpus, MAX_CPU_WORKERS)
    pin_memory: bool = True
    sampler: Optional[Sampler] = None


def collate_data_samples(batch: list[DataSample]) -> DataSample:
    input_combined = torch.stack([d.input for d in batch])
    if batch[0].target is not None:
        target_combined = torch.tensor(
            [d.target for d in batch],
            dtype=torch.long,
            requires_grad=False,
        )
    else:
        target_combined = None

    return DataSample(
        input=input_combined,
        target=target_combined,
    )


def load(
    dataset: Dataset,
    train: bool,
    config: DataLoaderConfig,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=train if config.sampler is None else False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        sampler=config.sampler,
        collate_fn=collate_data_samples,
    )


def get_data_type_loader(
    data: L.LightningDataModule,
    data_type: str,
) -> DataLoader:
    data.setup(data_type)
    loader: DataLoader
    if data_type == "train":
        loader = data.train_dataloader()
    elif data_type == "val":
        loader = data.val_dataloader()
    elif data_type == "test":
        loader = data.test_dataloader()
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    return loader
