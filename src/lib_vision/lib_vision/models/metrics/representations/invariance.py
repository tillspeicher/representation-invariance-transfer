from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, Union, cast

import numpy as np
import pandas as pd
import torch
import torchmetrics

from lib_dl_base.defs.task_id import TaskID
from lib_dl_base.io import persistence
from lib_vision.data.lib.multiclass_datamodule import MulticlassDataModule

from ...architectures.access import ModelConfig
from ...training import DeviceConfig
from .eval_on_representations import PU_LAYER_NAME, compute_representations
from .similarity.cka import cka_minibatch
from .similarity.exact import (
    cosine_similarity,
    dot_product_similarity,
    l2_distance,
)


DistanceMetric = Literal["dotprod", "cos", "l2", "cka"]


@dataclass
class InvarianceMeasurementConfig:
    # TODO: use list[DistanceMetric] when OmegaConf supports it
    metrics: list[str]
    shuffle_seed: int
    run: bool = True
    devices: DeviceConfig = field(default_factory=DeviceConfig)


def multi_measure_invariance(
    task_id: TaskID,
    config: InvarianceMeasurementConfig,
    models: dict[str, torch.nn.Module],
    model_configs: Union[ModelConfig, dict[str, ModelConfig]],
    datasets_1: dict[str, MulticlassDataModule],
    datasets_2: dict[str, MulticlassDataModule],
) -> dict[str, pd.DataFrame]:
    """
    datasets_1 and datasets_2 should only differ in their transformations.
    This allows us to compare how much the model's representations change
    based on the transformation applied to the objects, i.e. how invariant
    they are to the transformation.
    """
    assert list(datasets_1.keys()) == list(datasets_2.keys())
    shuffle_rng = np.random.default_rng(config.shuffle_seed)
    invariances = {
        metric: pd.DataFrame(
            columns=list(datasets_1.keys()),
            index=list(models.keys()),
            dtype=float,
        )
        for metric in config.metrics
    }
    for (data_name, dataset_1), dataset_2 in zip(
        datasets_1.items(), datasets_2.values()
    ):
        for model_name, model in models.items():
            print(
                f"Evaluating invariance for model {model_name} "
                f"on dataset {data_name}"
            )
            im_conf = deepcopy(config)
            im_conf.shuffle_seed = int(shuffle_rng.integers(0, 10000))
            model_conf: ModelConfig = (
                model_configs[model_name]
                if isinstance(model_configs, dict)
                else model_configs
            )
            model_invariance = measure_invariance(
                task_id=(task_id.set_model(model_name).set_dataset(data_name)),
                config=im_conf,
                model=model,
                model_config=model_conf,
                dataset_1=dataset_1,
                dataset_2=dataset_2,
            )
            for metric, metric_res in model_invariance.items():
                invariances[metric].loc[model_name, data_name] = metric_res
    return invariances


MEASUREMENT_ACTION = "invariance_measurement"
INVARIANCE_RESULT_NAME = "invariance_measurements"


def measure_invariance(
    task_id: TaskID,
    config: InvarianceMeasurementConfig,
    model: torch.nn.Module,
    model_config: ModelConfig,
    dataset_1: MulticlassDataModule,
    dataset_2: MulticlassDataModule,
) -> dict[str, float]:
    """Measure the invariance of a model to differences between two datasets

    Args:
        model: The model to measure the invariance of.
        dataset: The dataset producing tuples of identical inputs that
            just differ in their transformations.
        metric: The metric to use for measuring invariance. Can be either
            "dotprod" or "l2".

    Returns:
        A number representing the measured invariance of the model
    """
    if not config.run:
        return persistence.load(
            task_id.set_action(MEASUREMENT_ACTION),
            file_name=INVARIANCE_RESULT_NAME,
            obj_type=dict,
        )

    assert all(
        metric in ["dotprod", "cos", "l2", "cka"] for metric in config.metrics
    )
    metrics = cast(list[DistanceMetric], config.metrics)
    trackers = _CollectionTracker(metrics)
    shuffled_trackers = _CollectionTracker(metrics)
    _compute_rep_invariance(
        model,
        model_config,
        dataset_1,
        dataset_2,
        config.devices,
        trackers,
        shuffled_trackers,
        config.shuffle_seed,
    )
    same_transform_distances = trackers.compute()
    shuffled_normalization_distances = shuffled_trackers.compute()
    invariance_result = {
        metric: (
            float(
                same_transform_distances[metric]
                / shuffled_normalization_distances[metric]
            )
            if metric != "cka"
            # CKA is already normalized
            else float(same_transform_distances[metric])
        )
        for metric in metrics
    }
    # persistence.save(
    #     invariance_result,
    #     task_id.set_action(MEASUREMENT_ACTION),
    #     file_name=INVARIANCE_RESULT_NAME,
    # )
    return invariance_result


class _CollectionTracker:
    def __init__(self, metrics: list[DistanceMetric]) -> None:
        self.trackers = {}
        for metric in metrics:
            if metric == "cka":
                self.trackers[metric] = cka_minibatch.MinibatchCKA()
            else:
                self.trackers[metric] = _MetricTracker(metric)

    def update(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        for tracker in self.trackers.values():
            tracker.update(X, Y)

    def compute(self) -> dict[str, float]:
        return {
            metric: tracker.compute()
            for metric, tracker in self.trackers.items()
        }


class _MetricTracker(torchmetrics.Metric):
    def __init__(self, dist_metric: DistanceMetric) -> None:
        super().__init__()

        self.dist_metric: DistanceMetric = dist_metric
        self.dists: list[float]
        self.add_state("dists", default=[], dist_reduce_fx="cat")

    def update(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        mean_dist = _compute_metric(X, Y, self.dist_metric)
        self.dists.append(mean_dist)

    def compute(self) -> torch.Tensor:
        return torch.mean(torch.tensor(self.dists))


def _compute_rep_invariance(
    model: torch.nn.Module,
    model_config: ModelConfig,
    dataset_1: MulticlassDataModule,
    dataset_2: MulticlassDataModule,
    device_config: DeviceConfig,
    trackers: _CollectionTracker,
    shuffled_trackers: _CollectionTracker,
    shuffle_reps_seed: int,
) -> None:
    rep_iter_1 = compute_representations(
        model,
        dataset_1,
        device_config=device_config,
        model_config=model_config,
    )
    rep_iter_2 = compute_representations(
        model,
        dataset_2,
        device_config=device_config,
        model_config=model_config,
    )

    shuffle_rng = torch.Generator()
    torch.manual_seed(shuffle_reps_seed)
    try:
        while True:
            layer_reps_1 = next(rep_iter_1)
            pu_rep_1 = layer_reps_1[PU_LAYER_NAME]
            layer_reps_2 = next(rep_iter_2)
            pu_rep_2 = layer_reps_2[PU_LAYER_NAME]
            trackers.update(pu_rep_1, pu_rep_2)

            # Shuffle the representations to compute a normalization factor
            shuffled_pu_rep_1 = pu_rep_1[
                torch.randperm(pu_rep_1.shape[0], generator=shuffle_rng)
            ]
            shuffled_trackers.update(pu_rep_1, shuffled_pu_rep_1)

            del pu_rep_1
            del pu_rep_2
            torch.cuda.empty_cache()
    except StopIteration:
        pass


def _compute_metric(
    rep_1: torch.Tensor, rep_2: torch.Tensor, metric: DistanceMetric
) -> float:
    """Compute the invariance metric between two representations

    Args:
        rep_1: The first representation
        rep_2: The second representation
        metric: The metric to use for measuring invariance. Can be either
            "dotprod" or "l2".

    Returns:
        The row-wise metric results
    """
    if metric == "dotprod":
        value = dot_product_similarity(rep_1, rep_2)
    elif metric == "cos":
        value = cosine_similarity(rep_1, rep_2)
    elif metric == "l2":
        value = l2_distance(rep_1, rep_2)
    else:
        raise ValueError(f"Unknown metric {metric}")
    return value.detach().cpu().item()
