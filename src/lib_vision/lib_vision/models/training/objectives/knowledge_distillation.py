from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics

from lib_vision.data.wrappers.data_sample import DataSample
from lib_vision.utils import not_none

from ..optimization import OptimizerConfig


LOSS_METRIC = "loss"
AGREEMENT_METRIC = "agreement"

ClassInfo = Union[int, list[str]]


@dataclass
class KDConfig:
    kd_weight: float = 1.0
    temperature: float = 1.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    test_metrics: list[str] = field(default_factory=lambda: [])


class KnowledgeDistillation(L.LightningModule):
    def __init__(
        self,
        config: KDConfig,
        teacher_model: torch.nn.Module,
        student_model: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "config": config,
            }
        )

        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.student_model = student_model
        self.config = config
        self.train_agreement = AgreementMetric()
        self.val_agreement = AgreementMetric()
        self.test_agreement = AgreementMetric()

    def training_step(self, batch: DataSample) -> torch.Tensor:
        out_student, out_teacher = self._inference_step(batch)
        loss = _distillation_loss(
            out_student,
            out_teacher,
            not_none(batch.target),
            self.config.kd_weight,
            self.config.temperature,
        )
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
        )
        self.train_agreement(out_student, out_teacher)
        self.log(
            "train_agreement",
            self.train_agreement,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(
        self, batch: DataSample, batch_idx: int
    ) -> torch.Tensor:
        out_student, out_teacher = self._inference_step(batch)
        loss = _distillation_loss(
            out_student,
            out_teacher,
            not_none(batch.target),
            self.config.kd_weight,
            self.config.temperature,
        )
        self.log("val_loss", loss)
        self.val_agreement(out_student, out_teacher)
        self.log("val_agreement", self.val_agreement)
        return loss

    def test_step(self, batch: DataSample, batch_idx: int) -> None:
        out_student, out_teacher = self._inference_step(batch)
        self.test_agreement(out_student, out_teacher)

    def _inference_step(
        self,
        batch: DataSample,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = batch.input
        output_1 = self.student_model(x)
        # TODO: should we also set the model to eval-model?
        with torch.no_grad():
            # We don't want to update the parameters of the second (teacher)
            # model
            output_2 = self.teacher_model(x).detach()
        return output_1, output_2

    def configure_optimizers(self):
        if isinstance(self.student_model, L.LightningModule):
            return self.student_model.configure_optimizers()

        optimizer_config = self.config.optimizer
        optimizer = optimizer_config.optimizer(self.student_model)
        config_result: dict[str, Any] = {"optimizer": optimizer}
        if optimizer_config.lr_scheduler is not None:
            lr_scheduler = optimizer_config.lr_scheduler(optimizer)
            config_result["lr_scheduler"] = lr_scheduler
        return config_result


def _distillation_loss(
    out_teacher: torch.Tensor,
    out_student: torch.Tensor,
    target: torch.Tensor,
    kd_weight: float,
    temperature: float,
) -> torch.Tensor:
    supervised_loss = F.cross_entropy(out_student, target)
    kd_loss = F.kl_div(
        F.log_softmax(out_student / temperature, dim=1),
        F.softmax(out_teacher / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature**2)
    return supervised_loss + kd_weight * kd_loss


class AgreementMetric(torchmetrics.Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "num_matching", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state(
            "num_total", default=torch.tensor(0), dist_reduce_fx="sum"
        )

    def update(
        self, out_student: torch.Tensor, out_teacher: torch.Tensor
    ) -> None:
        assert out_student.shape == out_teacher.shape
        self.num_matching += torch.sum(
            torch.argmax(out_student, dim=1) == torch.argmax(out_teacher, dim=1)
        )
        self.num_total += out_student.shape[0]

    def compute(self) -> torch.Tensor:
        return self.num_matching.float() / self.num_total
