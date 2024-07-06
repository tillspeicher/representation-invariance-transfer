import itertools
from dataclasses import dataclass
from itertools import starmap
from typing import (
    Callable,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)

import lightning as L
import pandas as pd
import torch
from pandas.core.frame import functools

from lib_dl_base.defs.task_id import TaskID
from lib_vision.utils import not_none

from .tasks import DataInfo, ModelInfo
from .train import TrainingConfig, TrainingMetadata, TrainingResult


# (model_name, dataset_name)
ModelKey = tuple[str, str]
R = TypeVar("R")


@dataclass
class MultiResult(Generic[R]):
    """Result of a multi-training run. Results from different runs are
    indexed by (model, dataset) tuples."""

    results: dict[ModelKey, R]


MR = TypeVar("MR", bound="MultiResult")
AccessorResult = TypeVar("AccessorResult")
ItemAccessor = Callable[[MR], dict[ModelKey, AccessorResult]]
ModelKeyItem = Literal["model", "dataset"]


def with_single_key(accessor: ItemAccessor[MR, AccessorResult]):
    """Adds an accessor for only models or datasets, in addition the
    (model, dataset) tuples used by the MultiResult."""

    @overload
    def wrapper(
        self,
        keep_only: None = None,
    ) -> dict[ModelKey, AccessorResult]:
        ...

    @overload
    def wrapper(
        self,
        keep_only: ModelKeyItem,
    ) -> dict[str, AccessorResult]:
        ...

    @functools.wraps(accessor)
    def wrapper(
        self, keep_only: Optional[ModelKeyItem] = None
    ) -> Union[dict[ModelKey, AccessorResult], dict[str, AccessorResult]]:
        if keep_only is None:
            return accessor(self)

        if keep_only == "model":
            idx = 0
        elif keep_only == "dataset":
            idx = 1
        else:
            raise ValueError(f"Invalid key: {keep_only}")
        items = accessor(self)
        # Make sure the keys are unique
        assert len(items) == len(set(k[idx] for k in items))
        single_key_items = {k[idx]: v for k, v in items.items()}
        return single_key_items

    return wrapper


ModelSet = Union[ModelInfo, list[ModelInfo]]

D = TypeVar("D", bound=L.LightningDataModule)
TrainingData = Union[dict[str, D], tuple[str, D]]

TaskFunc = Callable[[TaskID, ModelInfo, DataInfo[D]], R]


def multi_exec(
    task_func: TaskFunc[D, R],
    description: TaskID,
    models: ModelSet,
    datasets: TrainingData[D],
    cross_product: bool = False,
    model_data_kwargs: dict[tuple[str | ModelKey, str], dict] = {},
    **kwargs,
) -> MultiResult[R]:
    if isinstance(models, list):
        model_iter = models
    else:
        model_iter = [models]
    if isinstance(datasets, tuple):
        dataset_iter = [datasets]
    else:
        dataset_iter = list(datasets.items())
    if cross_product:
        get_combined_iter = lambda: itertools.product(  # noqa E731
            model_iter, dataset_iter
        )
    else:
        assert len(model_iter) == len(dataset_iter)
        get_combined_iter = lambda: zip(model_iter, dataset_iter)  # noqa E731

    # We use starmap to potentially parallelize the execution later
    training_results = cast(
        list[R],
        starmap(
            _exec_wrapper,
            [
                (
                    task_func,
                    description,
                    model,
                    dataset_name,
                    dataset,
                    # training_config,
                    kwargs
                    | model_data_kwargs.get((model.name, dataset_name), {}),
                )
                for model, (dataset_name, dataset) in get_combined_iter()
            ],
        ),
    )
    results_map = {
        (model_info.name, dataset_name): result
        for (model_info, (dataset_name, _)), result in zip(
            get_combined_iter(), training_results
        )
    }
    return MultiResult(
        results=results_map,
    )


def _exec_wrapper(
    func: TaskFunc[D, R],
    description: TaskID,
    model: ModelInfo,
    dataset_name: str,
    dataset: D,
    kwargs,
) -> R:
    return func(
        description.set_model(model.name).set_dataset(dataset_name),
        model,
        DataInfo(dataset_name, dataset),
        **kwargs,
    )


@dataclass
class MultiTrainingResult(MultiResult[TrainingResult]):
    @with_single_key
    def models(
        self,
    ) -> dict[ModelKey, torch.nn.Module]:
        assert all(
            model_result.model is not None
            for model_result in self.results.values()
        ), "Some models are None"
        return {
            model_key: cast(torch.nn.Module, model_result.model)
            for model_key, model_result in self.results.items()
        }

    @with_single_key
    def metadata(self) -> dict[ModelKey, TrainingMetadata]:
        return {
            model_key: model_result.metadata
            for model_key, model_result in self.results.items()
        }

    @property
    def metrics(self) -> pd.DataFrame:
        metric_results = [
            not_none(model_results.metrics)
            for model_results in self.results.values()
        ]
        metrics = pd.concat(
            metric_results,
            keys=list(self.results.keys()),
        )
        metrics.index.names = ["model", "dataset", "split"]
        return metrics

    def without_models(self) -> "MultiTrainingResult":
        """Removes the models from the result object, which makes it
        possible to store it independently from the models
        (which are already stored with checkpoints)."""
        return MultiTrainingResult(
            results={
                model_key: model_result.without_model()
                for model_key, model_result in self.results.items()
            }
        )


TrainingTaskFunc = Callable[[TaskID, ModelInfo, D, TrainingConfig], R]


def multi_training(
    task_func: TrainingTaskFunc[D, TrainingResult],
    description: TaskID,
    models: ModelSet,
    datasets: TrainingData[D],
    training_config: TrainingConfig,
    cross_product: bool = False,
    model_data_kwargs: dict[tuple[str | ModelKey, str], dict] = {},
    **kwargs,
) -> MultiTrainingResult:
    """Executes a training task for multiple models and datasets."""
    return MultiTrainingResult(
        results=multi_exec(
            # TODO: fix type issue
            cast(TaskFunc, task_func),
            description,
            models,
            datasets,
            training_config=training_config,
            cross_product=cross_product,
            model_data_kwargs=model_data_kwargs,
            **kwargs,
        ).results,
    )
