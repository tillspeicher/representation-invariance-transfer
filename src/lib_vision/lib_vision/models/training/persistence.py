from dataclasses import dataclass
from typing import Optional

import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from lib_dl_base.defs.task_id import TaskID
from lib_dl_base.io import persistence as results_persistence

from ..io.logging import load_csv_log
from ..io.persistence import models as models_persistence


@dataclass
class TrainingMetadata:
    task_id: TaskID
    training_time: float

    def __post_init__(self):
        self._logs = None

    @property
    def logs(self) -> pd.DataFrame:
        if self._logs is None:
            self._logs = load_csv_log(self.task_id)
        return self._logs


@dataclass
class TrainingResult:
    metadata: TrainingMetadata
    model: Optional[torch.nn.Module]
    metrics: Optional[pd.DataFrame] = None

    def without_model(self) -> "TrainingResult":
        return TrainingResult(
            metadata=self.metadata,
            model=None,
            metrics=self.metrics,
        )


TRAINING_METADATA_NAME = "training_metadata"


# TODO: make sure we don't save the model in case the full training result
# is passed in here
def save_training_metadata(
    metadata: TrainingMetadata,
) -> None:
    # We pass in a TrainingResult here including the model, so we need to
    # prune it before saving
    clean_metadata = TrainingMetadata(
        task_id=metadata.task_id,
        training_time=metadata.training_time,
    )
    results_persistence.save(
        clean_metadata,
        metadata.task_id,
        file_name=TRAINING_METADATA_NAME,
    )


_DUMMY_MODEL = torch.nn.Sequential()


def load_training_result(
    task_id: TaskID,
    task: L.LightningModule,
    require_model: bool = True,
) -> TrainingResult:
    if require_model:
        model = models_persistence.load_model(
            task_id,
            task,
        )
    else:
        model = _DUMMY_MODEL
    training_metadata = results_persistence.load(
        task_id,
        file_name=TRAINING_METADATA_NAME,
        obj_type=TrainingMetadata,
    )
    return TrainingResult(
        metadata=training_metadata,
        model=model,
    )


def plot_training_stat(
    logs: pd.DataFrame,
    *stat_names: str,
) -> None:  # plt.Figure:
    fig, ax = plt.subplots(1, figsize=(5, 4), squeeze=False)
    stat_plot = ax[0][0]
    for stat_name in stat_names:
        data = logs[stat_name].dropna().reset_index(drop=True)
        sns.lineplot(
            data=data,
            label=stat_name,
            axes=stat_plot,
        )
    stat_plot.set_xlabel("Epoch")
    fig.tight_layout()
    plt.show(fig)
