import pandas as pd

from lib_dl_base.defs.task_id import TaskID
from lib_vision.models.training.multi_training import (
    MultiTrainingResult,
    TrainingMetadata,
    TrainingResult,
)


def test_multi_training_result_metrics():
    description = TaskID("test", "model", "data")
    index = pd.Index(["test"], name="split")
    results = {
        ("model_1", "data_1"): TrainingResult(
            metadata=TrainingMetadata(
                task_id=description,
                training_time=1.0,
            ),
            model=None,
            metrics=pd.DataFrame({"accuracy": [0.5]}, index=index),
        ),
        ("model_2", "data_1"): TrainingResult(
            metadata=TrainingMetadata(
                task_id=description,
                training_time=2.0,
            ),
            model=None,
            metrics=pd.DataFrame({"accuracy": [0.6]}, index=index),
        ),
    }
    multi_training_result = MultiTrainingResult(results)
    metrics = multi_training_result.metrics

    index = pd.MultiIndex.from_tuples(
        [
            ("model_1", "data_1", "test"),
            ("model_2", "data_1", "test"),
        ],
        names=["model", "dataset", "split"],
    )
    expected_metrics = pd.DataFrame(
        {"accuracy": [0.5, 0.6]},
        index=index,
    )
    pd.testing.assert_frame_equal(metrics, expected_metrics)
