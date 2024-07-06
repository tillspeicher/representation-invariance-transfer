from typing import Iterator, Optional

import lightning as L
import torch

from ...architectures.access import ModelConfig
from ...modifiers.intermediate_monitoring import (
    get_layers_to_monitor,
    intermediate_representations,
)
from ...training import DeviceConfig


PU_LAYER_NAME = "pu"


def compute_representations(
    model: torch.nn.Module,
    dataset: L.LightningDataModule,
    device_config: DeviceConfig,
    model_config: Optional[ModelConfig] = None,
    layer_names: Optional[list[str]] = None,
) -> Iterator[dict[str, torch.Tensor]]:
    pu_layer_name_to_append = None
    layer_names_to_monitor = get_layers_to_monitor(layer_names, model_config)
    if layer_names_to_monitor != layer_names:
        pu_layer_name_to_append = layer_names_to_monitor[0]

    dataset.setup("test")
    test_loader = dataset.test_dataloader()

    # model = deepcopy(model)
    model.to(device_config.accelerator)
    model.eval()
    with intermediate_representations(
        model,
        layer_names_to_monitor,
        flatten=True,
    ) as monitored_model:
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                # TODO: somehow the test dataloader stops after 1 batch
                # so we use this workaround. Investigate and fix.
                # n_loaded_points = 0
                # while n_loaded_points < n_total_points:
                # batch = next(test_loader)
                x = batch.input.to(device_config.accelerator)
                # n_loaded_points += len(x)
                output = monitored_model(x)
                yield _convert_representations(
                    output.layer_reps,
                    pu_layer_name_to_append,
                )
    del model
    torch.cuda.empty_cache()


def _convert_representations(
    layer_reps: dict[str, torch.Tensor],
    penultimate_layer_name: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    if penultimate_layer_name is not None:
        layer_reps[PU_LAYER_NAME] = layer_reps[penultimate_layer_name]
    return layer_reps
