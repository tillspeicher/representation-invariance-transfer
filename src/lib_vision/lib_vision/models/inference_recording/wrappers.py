import torch

from .inference_record import InferenceRecord, to_inference_record


def with_wrappers(model: torch.nn.Module) -> torch.nn.Module:
    source_forward = model.forward

    # TODO: annotate batch type here
    def wrapped_forward(batch) -> InferenceRecord:
        output = source_forward(batch.input)
        return to_inference_record(output)

    model.forward = wrapped_forward
    return model
