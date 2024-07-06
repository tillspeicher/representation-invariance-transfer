from contextlib import contextmanager
from typing import Callable, Iterator, Optional

import torch

from ..architectures import access
from ..architectures.access import ModelConfig
from ..inference_recording import InferenceRecord, to_inference_record


@contextmanager
def intermediate_representations(
    model: torch.nn.Module,
    layers: list[str],
    flatten: bool = True,
) -> Iterator[torch.nn.Module]:
    monitored_model = IntermediateMonitoringModel(model, layers, flatten)
    yield monitored_model
    monitored_model.remove_hooks()


class IntermediateMonitoringModel(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        layers: list[str],
        flatten: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.layers = layers
        self.flatten = flatten

        self.layer_reps = {}
        self.hooks = []

        def activation_monitor_hook(
            layer_name: str,
        ) -> Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], None]:
            def hook(
                model: torch.nn.Module,
                input: torch.Tensor,
                output: torch.Tensor,
            ) -> None:
                self.layer_reps[layer_name] = output  # .detach()

            return hook

        for layer_name in layers:
            layer = access.get_layer(model, layer_name)
            hook = layer.register_forward_hook(
                activation_monitor_hook(layer_name)  # type: ignore
            )
            self.hooks.append(hook)

    def forward(self, x: torch.Tensor) -> InferenceRecord:
        self.layer_reps.clear()
        output = to_inference_record(self.model(x))
        if self.flatten:
            layer_reps = {
                layer_name: torch.flatten(reps, start_dim=1)
                for layer_name, reps in self.layer_reps.items()
            }
        else:
            layer_reps = self.layer_reps.copy()
        return InferenceRecord(
            output=output.output,
            layer_reps=layer_reps,
        )

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()


def get_layers_to_monitor(
    layers: Optional[list[str]],
    model_config: Optional[ModelConfig],
) -> list[str]:
    if layers is None:
        assert model_config is not None
        model_pu_layer_name = access.get_penultimate_layer_name(model_config)
        return [model_pu_layer_name]
    else:
        return layers
