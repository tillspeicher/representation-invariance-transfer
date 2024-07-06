from dataclasses import dataclass
from typing import Iterable, Optional, Union

import torch


"""The representations at different layers of a network.
"""
LayerRepresentations = dict[str, torch.Tensor]


@dataclass
class InferenceRecord:
    """Records the results of an inference pass through the model.

    Attributes
    ----------
    output: torch.Tensor
        The final-layer logit activations.
    layer_reps : LayerRepresentations, optional
        Records of the activations of intermediate layers in the network.
    """

    output: torch.Tensor
    layer_reps: Optional[LayerRepresentations] = None

    """Computes the indices of the predicted classes.

    Returns
    -------
    torch.Tensor
        For each input in the batch the index with the largest
        logit activations.
    """

    def predictions(self) -> torch.Tensor:
        return torch.max(self.output, 1)[1]

    def __len__(self) -> int:
        return len(self.output)


def to_inference_record(
    output: Union[torch.Tensor, InferenceRecord],
) -> InferenceRecord:
    if isinstance(output, InferenceRecord):
        return output
    else:
        return InferenceRecord(output=output)


def concatenate(maps: Iterable[LayerRepresentations]) -> LayerRepresentations:
    """Combines multiple layer representations horizontally into one.

    Useful for combining layer represenations from individual elements in a
    batch into one.
    """
    layer_names = None
    for rep_map in maps:
        if layer_names is None:
            layer_names = rep_map.keys()
        else:
            if rep_map.keys() != layer_names:
                raise ValueError(
                    "Representation maps need to have the same layers"
                )

    if layer_names is None:
        raise ValueError("You must provide at least one map")
    return {
        layer: torch.cat([rep_map[layer] for rep_map in maps])
        for layer in layer_names
    }
