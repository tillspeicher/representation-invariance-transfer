from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Union, cast

import lightning as L
import torch
import torchmetrics

from lib_vision.data.wrappers.data_sample import DataSample
from lib_vision.models.inference_recording import InferenceRecord
from lib_vision.models.metrics.representations.similarity import (
    SimilarityMetric,
    cosine_similarity,
    dot_product_similarity,
    l2_distance,
)
from lib_vision.models.metrics.representations.similarity.cka import (
    MinibatchCKA,
    linear_CKA_torch,
)
from lib_vision.models.modifiers.intermediate_monitoring import (
    IntermediateMonitoringModel,
    ModelConfig,
    get_layers_to_monitor,
)

from ..persistence import TrainingResult


@dataclass
class RepSimilarityTrainingResult(TrainingResult):
    model_2: Optional[torch.nn.Module] = None


RepSimFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
EvalMetric = Literal["l2", "cos", "cka"]
TrainingMetric = Union[EvalMetric, RepSimFunction]
# if single, train the first model using the representations of the
# second model as targets
ModelToUpdate = Literal["single", "both"]


@dataclass
class RepSimilarityConfig:
    similarity_metric: EvalMetric  # TrainingMetric
    model_to_update: ModelToUpdate
    eval_metrics: list[EvalMetric] = field(default_factory=lambda: ["cka"])
    model_1_layer: Optional[str] = None
    model_1_config: Optional[ModelConfig] = None
    model_2_layer: Optional[str] = None
    model_2_config: Optional[ModelConfig] = None
    metric_func_override: Optional[RepSimFunction] = None


class RepSimLearning(L.LightningModule):
    """A class to train models directly for representation similarity"""

    def __init__(
        self,
        config: RepSimilarityConfig,
        # Student model
        model_1: torch.nn.Module,
        # Teacher model
        model_2: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "config": config,
            }
        )
        self.automatic_optimization = False

        self.config = config
        # Student model
        self.model_1, self.model_1_layer = RepSimLearning._setup_model(
            model_1, config.model_1_layer, config.model_1_config
        )
        # Teacher model
        self.model_2, self.model_2_layer = RepSimLearning._setup_model(
            model_2, config.model_2_layer, config.model_2_config
        )

        metrics = RepSimLearning._get_metrics(
            cast(Any, config.eval_metrics), config.similarity_metric
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="")  # prefix="test_")

    def training_step(
        self,
        batch: DataSample,
        batch_idx: int,
    ) -> Optional[torch.Tensor]:
        if self.config.model_to_update == "single":
            freeze_1 = False
            freeze_2 = True
        elif self.config.model_to_update == "both":
            freeze_1 = False
            freeze_2 = False
        else:
            raise ValueError()
        output_1 = self._inference_step(self.model_1, batch, freeze=freeze_1)
        output_2 = self._inference_step(self.model_2, batch, freeze=freeze_2)
        reps_1 = self._get_representations(
            output_1, self.model_1_layer, freeze=freeze_1
        )
        reps_2 = self._get_representations(
            output_2, self.model_2_layer, freeze=freeze_2
        )
        loss = self._compute_loss(reps_1, reps_2)

        self.manual_backward(loss)
        optimizers = self.optimizers()
        assert isinstance(optimizers, list)
        freezes = [freeze_1, freeze_2]
        for optimizer, freeze_model in zip(optimizers, freezes):
            if not freeze_model:
                optimizer.step()
                optimizer.zero_grad()

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
        )
        self.train_metrics.update(reps_1, reps_2)
        # self._log_representation_metrics(reps_1, reps_2, "train")
        # self._log_output_metrics(output_1.output, output_2.output, "train")
        # return loss
        return None

    def on_training_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(
        self, batch: DataSample, batch_idx: int
    ) -> torch.Tensor:
        output_1 = self._inference_step(self.model_1, batch)
        output_2 = self._inference_step(self.model_2, batch)
        reps_1 = self._get_representations(output_1, self.model_1_layer)
        reps_2 = self._get_representations(output_2, self.model_2_layer)
        loss = self._compute_loss(reps_1, reps_2)

        self.log("val_loss", loss)
        self.val_metrics.update(reps_1, reps_2)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, batch: DataSample, batch_idx: int) -> None:
        output_1 = self._inference_step(self.model_1, batch)
        output_2 = self._inference_step(self.model_2, batch)
        reps_1 = self._get_representations(output_1, self.model_1_layer)
        reps_2 = self._get_representations(output_2, self.model_2_layer)
        self.test_metrics.update(reps_1, reps_2)

    # def test_epoch_end(self, _) -> None:
    #     self.log_dict(self.test_metrics.compute())
    #     self.test_metrics.reset()

    def _inference_step(
        self,
        model: torch.nn.Module,
        batch: DataSample,
        freeze: bool = False,
    ) -> InferenceRecord:
        x = batch.input
        if freeze:
            model.eval()
            with torch.no_grad():
                output = model(x)
        else:
            model.train()
            output = model(x)
        # output = model(x)
        return output

    def _get_representations(
        self,
        output: InferenceRecord,
        layer_name: str,
        freeze: bool = False,
    ) -> torch.Tensor:
        assert output.layer_reps is not None
        reps = output.layer_reps[layer_name]
        if freeze:
            return reps.detach()
        else:
            return reps

    def _compute_loss(
        self, reps_1: torch.Tensor, reps_2: torch.Tensor
    ) -> torch.Tensor:
        if self.config.metric_func_override is not None:
            loss = self.config.metric_func_override(reps_1, reps_2)
        elif self.config.similarity_metric == "l2":
            loss = l2_distance(reps_1, reps_2)
            # return F.mse_loss(reps_1, reps_2)
        elif self.config.similarity_metric == "dot":
            loss = -dot_product_similarity(reps_1, reps_2)
            # return -torch.sum(reps_1 * reps_2, -1).mean()
        elif self.config.similarity_metric == "cos":
            loss = -cosine_similarity(reps_1, reps_2)
        elif self.config.similarity_metric == "cka":
            # Minimize negative CKA similarity, i.e. maximize similarity
            loss = -linear_CKA_torch(reps_1, reps_2)
        else:
            raise ValueError(
                f"Unknown similarity metric {self.config.similarity_metric}"
            )
        return loss

    # TODO: make this configurable
    def configure_optimizers(self):
        learning_rate = 0.001
        return (
            torch.optim.Adam(self.model_1.parameters(), lr=learning_rate),
            torch.optim.Adam(self.model_2.parameters(), lr=learning_rate),
        )
        # if self.config.model_to_update == "single":
        #     return (
        #         torch.optim.Adam(self.model_1.parameters(), lr=learning_rate),
        #     )
        # else:
        #     return (
        #         torch.optim.Adam(self.model_1.parameters(), lr=learning_rate),
        #         torch.optim.Adam(self.model_2.parameters(), lr=learning_rate),
        #     )

    @staticmethod
    def _setup_model(
        model: torch.nn.Module,
        layer_name: Optional[str],
        model_config: Optional[ModelConfig],
    ) -> tuple[torch.nn.Module, str]:
        monitored_layer = (
            [layer_name]
            if layer_name is not None
            else get_layers_to_monitor(None, model_config)
        )[0]
        monitored_model = IntermediateMonitoringModel(
            model,
            [monitored_layer],
            flatten=True,
        )
        return monitored_model, monitored_layer

    @staticmethod
    def _get_metrics(
        metric_names: list[str],
        loss: TrainingMetric,
    ) -> torchmetrics.MetricCollection:
        metrics = {}
        for metric_name in metric_names:
            if metric_name == "cka":
                metrics["cka"] = MinibatchCKA()
            elif metric_name == "l2":
                metrics["l2"] = SimilarityMetric(l2_distance)
            elif metric_name == "cos":
                metrics["cos"] = SimilarityMetric(cosine_similarity)
            else:
                raise ValueError(f"Unknown metric {metric_name}")
        if loss not in metrics and isinstance(loss, Callable):
            metrics["loss"] = SimilarityMetric(loss)
        return torchmetrics.MetricCollection(metrics)
