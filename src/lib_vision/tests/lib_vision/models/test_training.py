import os
from pathlib import Path
from typing import Optional

# from torchvision import transforms
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from lib_dl_base.defs.task_id import TaskID
from lib_vision.data.loading import DataLoaderConfig, load
from lib_vision.data.wrappers.data_sample import DataSample
from lib_vision.models.training import (
    DeviceConfig,
    Trainer,
    TrainingConfig,
    train,
)
from lib_vision.models.training.objectives.supervised import SupervisedLearning


NUM_SAMPLES = 64


class DummyDataset(Dataset):
    NUM_CLASSES = 2

    def __len__(self) -> int:
        return NUM_SAMPLES

    def __getitem__(self, idx: int) -> DataSample:
        # x = torch.rand((3, 32, 32))
        x = torch.rand((10,))
        y = torch.randint(low=0, high=self.NUM_CLASSES, size=(1,))
        return DataSample(
            input=x,
            target=y,
        )


class DummyDataModule(L.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = DummyDataset()
        self.loader_config = DataLoaderConfig(
            batch_size=NUM_SAMPLES,
        )

    def train_dataloader(self) -> DataLoader:
        return load(self.dataset, train=True, config=self.loader_config)

    def val_dataloader(self) -> DataLoader:
        return load(self.dataset, train=False, config=self.loader_config)

    def test_dataloader(self) -> DataLoader:
        return load(self.dataset, train=False, config=self.loader_config)


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_training_manual(tmp_path: Path):
    # Define a Pytorch Lightning Module with all necessary configuration
    # Seehttps://pytorch-lightning.readthedocs.io/en/stable/common
    # /lightning_module.html
    os.environ["ARTIFACTS_DIR"] = str(tmp_path / "artifacts")
    model = DummyModel()
    training_task = SupervisedLearning(
        model=model,
        classes=DummyDataset.NUM_CLASSES,
    )

    dataset = DummyDataset()
    dataloader = load(
        dataset,
        train=True,
        config=DataLoaderConfig(
            batch_size=NUM_SAMPLES,
            num_workers=1,
        ),
    )

    # Create a trainer and fit the model
    trainer = Trainer(
        task_id=TaskID(
            task_prefixes=["test", "training", f"sid_{0}"],
        ),
        devices=DeviceConfig(
            accelerator="cpu",
            devices=[0],
        ),
        max_epochs=1,
        limit_train_batches=1,
    )
    trainer.fit(training_task, dataloader)


def test_training_tooling(tmp_path: Path):
    os.environ["ARTIFACTS_DIR"] = str(tmp_path / "artifacts")
    model = DummyModel()
    training_task = SupervisedLearning(
        model=model,
        classes=DummyDataset.NUM_CLASSES,
        test_metrics=["accuracy"],
    )
    dataset = DummyDataModule()
    training_config = TrainingConfig(
        max_epochs=1,
        max_steps=1,
        save_checkpoints=False,
        train=True,
        eval=True,
        devices=DeviceConfig(
            accelerator="cpu",
            devices=[0],
        ),
    )
    _ = train(
        task_id=TaskID(
            task_prefixes=["test", "training", f"sid_{0}"],
        ),
        data=dataset,
        config=training_config,
        training_task=training_task,
        eval_task=training_task,
    )
