from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import classification

from ...inference_recording import to_inference_record
from ..optimization import OptimizerConfig


ACCURACY_METRIC = "accuracy"
LOSS_METRIC = "loss"
CLASSWISE_ACCURACY_METRIC = "classwise_accuracy"
CONFUSION_MATRIX_METRIC = "confusion_matrix"

ClassInfo = Union[int, list[str]]


# class SupervisedLearning(L.LightningModule):
class SupervisedLearning(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        classes: ClassInfo,
        optimizer_config: OptimizerConfig = OptimizerConfig(),
        test_metrics: list[str] = [],
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                # "model": model,
                "classes": classes,
                "optimizer_config": optimizer_config,
                "test_metrics": test_metrics,
            }
        )

        self.model = model
        self.optimizer_config = optimizer_config
        # self.test_metrics = test_metrics
        self.train_accuracy = _create_metric(ACCURACY_METRIC, classes)
        self.val_accuracy = _create_metric(ACCURACY_METRIC, classes)

        if ACCURACY_METRIC not in test_metrics:
            test_metrics = [ACCURACY_METRIC] + test_metrics
        self.test_metrics = torchmetrics.MetricCollection(
            {
                metric_name: _create_metric(metric_name, classes)
                for metric_name in test_metrics
            }
        )

    # TODO: annotate the batch type
    def training_step(self, batch) -> torch.Tensor:
        output = to_inference_record(self.model(batch.input))
        loss = F.cross_entropy(output.output, batch.target)
        self.log(
            "train_loss",
            # "loss/train",
            loss,
            on_step=False,
            on_epoch=True,
        )

        self.train_accuracy(output.output, batch.target)
        self.log(
            # "accuracy/train",
            "train_accuracy",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        # x, y = batch.x, batch.y
        output = to_inference_record(self.model(batch.input))
        loss = F.cross_entropy(output.output, batch.target)
        # self.log("loss/val", loss)
        self.log("val_loss", loss)

        self.val_accuracy(output.output, batch.target)
        self.log("val_accuracy", self.val_accuracy)
        return loss

    def test_step(self, batch, batch_idx: int) -> None:
        output = to_inference_record(self.model(batch.input))
        self.test_metrics(output.output, batch.target)
        # self.log("testmetrics", self.test_metrics)
        # self.test_metrics(output.output, batch.target)
        # self.log("testmetrics", self.test_metrics(output.output, batch.target))

    def configure_optimizers(self):
        if isinstance(self.model, L.LightningModule):
            return self.model.configure_optimizers()

        optimizer = self.optimizer_config.optimizer(self.model)
        config_result: dict[str, Any] = {"optimizer": optimizer}
        if self.optimizer_config.lr_scheduler is not None:
            lr_scheduler = self.optimizer_config.lr_scheduler(optimizer)
            config_result["lr_scheduler"] = lr_scheduler
        return config_result


def _create_metric(
    metric_name: str,
    classes: Union[int, list[str], None] = None,
) -> torchmetrics.Metric:
    if classes is None:
        # num_classes = None
        raise ValueError("You need to specify the number of classes")
    else:
        num_classes = classes if isinstance(classes, int) else len(classes)
    if metric_name == ACCURACY_METRIC:
        return classification.MulticlassAccuracy(num_classes=num_classes)
    elif metric_name == CLASSWISE_ACCURACY_METRIC:
        metric = classification.MulticlassAccuracy(
            task="multiclass", average="none", num_classes=num_classes
        )
    elif metric_name == CONFUSION_MATRIX_METRIC:
        metric = torchmetrics.ConfusionMatrix(
            task="multiclass",
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unsupported metric {metric_name}")
    if isinstance(classes, list):
        metric = torchmetrics.ClasswiseWrapper(metric, classes)
    return metric


def get_dataset_classes(
    classes: Union[ClassInfo, dict[str, ClassInfo]],
    data_name: str,
) -> ClassInfo:
    if isinstance(classes, dict):
        data_classes = classes[data_name]
        if isinstance(data_classes, list):
            return data_classes
        else:
            return data_classes
    else:
        return classes


def get_num_dataset_classes(
    classes: Union[ClassInfo, dict[str, ClassInfo]],
    data_name: str,
) -> int:
    dataset_classes = get_dataset_classes(classes, data_name)
    return (
        len(dataset_classes)
        if isinstance(dataset_classes, list)
        else dataset_classes
    )
