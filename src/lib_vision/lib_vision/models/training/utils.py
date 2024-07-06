import torch

from .persistence import TrainingResult
from .objectives.supervised import SupervisedLearning
from .objectives.rep_similarity import (
    RepSimilarityTrainingResult,
    RepSimLearning,
)
from .objectives.knowledge_distillation import KnowledgeDistillation


def extract_model(result: TrainingResult) -> TrainingResult:
    if result.model is None:
        return result
    elif isinstance(result.model, SupervisedLearning):
        extracted_model = result.model.model
    elif isinstance(result.model, KnowledgeDistillation):
        extracted_model = result.model.student_model
    elif isinstance(result.model, RepSimLearning):
        # assert result.model.config.model_to_update == "single"
        # Extract the model from the IntermediateMonitoringModel
        extracted_model = result.model.model_1.model
        model_2 = result.model.model_2.model
        return RepSimilarityTrainingResult(
            metadata=result.metadata,
            model=extracted_model,
            model_2=model_2,
            metrics=result.metrics,
        )
    else:
        raise NotImplementedError()
    return TrainingResult(
        metadata=result.metadata,
        model=extracted_model,
        metrics=result.metrics,
    )


def disable_gradients(model: torch.nn.Module) -> torch.nn.Module:
    for param in model.parameters():
        param.requires_grad = False
    # Set the previous layers to eval mode to prevent the BatchNorm
    # statistics from updating during fine-tuning
    for module in model.modules():
        module.eval()
    return model


def enable_gradients(model: torch.nn.Module) -> torch.nn.Module:
    for param in model.parameters():
        param.requires_grad = True
    for module in model.modules():
        module.train()
    return model
