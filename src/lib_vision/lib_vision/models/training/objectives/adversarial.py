from dataclasses import dataclass, asdict
from typing import Literal, Optional
import torch
from robustness.attacker import AttackerModel

from models.inference_recording import InferenceRecord


AttackConstraint = Literal["2", "inf", "unconstrained", "fourier"]


@dataclass
class AdversarialTrainingConfig:
    constraint: AttackConstraint
    eps: float
    step_size: float
    iterations: int
    random_start: bool = False
    random_restarts: bool = False


@dataclass
class DatasetStats:
    mean: torch.Tensor
    std: torch.Tensor


@dataclass
class AdvInferenceRecord(InferenceRecord):
    adv_input: Optional[torch.Tensor] = None


class AdversarialTraining(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset_stats: DatasetStats,
        config: AdversarialTrainingConfig,
    ) -> None:
        super().__init__()

        self.config = config

        orig_forward = model.forward

        def compat_forward(
            x,
            *,
            with_latent: bool = False,
            fake_relu: bool = False,
            no_relu: bool = False,
        ):
            """Forward-function replacement to be compatible with robustness
            library"""
            return orig_forward(x)

        model.forward = compat_forward

        self.model = model
        self.attacker = AttackerModel(model, dataset_stats)

    def forward(
        self,
        # TODO: annotate BatchSample type
        batch,
        adversarial: bool = True,
    ) -> AdvInferenceRecord:
        # x = batch.input.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            output, adv_input = self.attacker(
                inp=batch.input,
                # inp=x,
                target=batch.target,
                make_adv=adversarial,
                **asdict(self.config),
            )
        if isinstance(output, InferenceRecord):
            return AdvInferenceRecord(
                **asdict(output),
                adv_input=adv_input,
            )
        else:
            return AdvInferenceRecord(
                output=output,
                adv_input=adv_input,
            )
