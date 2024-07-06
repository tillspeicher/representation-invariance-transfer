import json
from typing import Iterable

import pytest

from lib_dl_base.io.dirs import get_artifacts_dir, set_artifacts_dir
from lib_project.experiment import (
    ExperimentConfig,
    ExperimentID,
    ExperimentResult,
    experiment,
    iterative_experiment,
)


@pytest.fixture
def artifacts_dir(tmp_path):
    set_artifacts_dir(tmp_path / "artifacts")
    return get_artifacts_dir()


def test_return_experiment(artifacts_dir):
    expected_experiment_id = ExperimentID(
        experiment_name="return_experiment_name",
        config_name="return_config_name",
        seed_id=0,
    )

    @experiment("return_experiment_name")
    def return_test_experiment(
        config: ExperimentConfig,
        experiment_id: ExperimentID,
    ) -> dict[str, str]:
        assert config.name == "return_config_name"
        assert config.seed_id == 0
        assert experiment_id.name == expected_experiment_id.name

        return {
            "value": "test",
        }

    config = ExperimentConfig(
        name="return_config_name",
        seed_id=0,
    )
    res = return_test_experiment(config)
    assert isinstance(res, ExperimentResult)
    assert res.config == config
    assert res.value == {
        "value": "test",
    }

    result_path = artifacts_dir / expected_experiment_id.path / "result.json"
    assert result_path.exists()
    with result_path.open("r") as f:
        loaded_result = json.load(f)
        assert isinstance(loaded_result["execution_time"], float)
        del loaded_result["execution_time"]
        assert loaded_result == {
            "config": {
                "name": "return_config_name",
                "seed_id": 0,
                "group": None,
            },
            "value": {
                "value": "test",
            },
        }


def test_iterative_experiment(artifacts_dir):
    expected_experiment_id = ExperimentID(
        experiment_name="iterative_experiment_name",
        config_name="iterative_config_name",
        seed_id=0,
    )

    @iterative_experiment("iterative_experiment_name")
    def iterative_test_experiment(
        config: ExperimentConfig,
        experiment_id: ExperimentID,
    ) -> Iterable[dict[str, str]]:
        assert config.name == "iterative_config_name"
        assert config.seed_id == 0
        assert experiment_id.name == expected_experiment_id.name

        for i in range(3):
            yield {
                "value": f"test_{i}",
            }

    config = ExperimentConfig(
        name="iterative_config_name",
        seed_id=0,
    )
    result_path = artifacts_dir / expected_experiment_id.path / "result.json"

    res_iter = iterative_test_experiment(config)
    for i, res in enumerate(res_iter):
        assert isinstance(res, ExperimentResult)
        assert res.config == config
        assert res.value == {
            "value": f"test_{i}",
        }

        assert result_path.exists()
        with result_path.open("r") as f:
            loaded_result = json.load(f)
            assert isinstance(loaded_result["execution_time"], float)
            del loaded_result["execution_time"]
            assert loaded_result == {
                "config": {
                    "name": "iterative_config_name",
                    "seed_id": 0,
                    "group": None,
                },
                "value": {
                    "value": f"test_{i}",
                },
            }
