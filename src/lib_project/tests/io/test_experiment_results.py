from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lib_dl_base.io.dirs import set_artifacts_dir
from lib_project.io.experiment_results import (
    ExperimentConfig,
    ExperimentResult,
    NoSave,
    NoSaveValue,
    TaskID,
    load_result,
    save_result,
)


def test_save_load_result(tmp_path: Path):
    res = ExperimentResult(
        config=ExperimentConfig(
            name="test",
            seed_id=0,
        ),
        execution_time=1.0,
        value=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
    )
    task_id = TaskID(
        action="test_action",
        model="test_model",
        dataset="test_ds",
    )
    set_artifacts_dir(tmp_path / "artifacts")

    save_result(task_id=task_id, result=res)
    loaded_res = load_result(ExperimentConfig, pd.DataFrame, task_id=task_id)
    assert res.__repr__() == loaded_res.__repr__()
    for k, v in res.__dict__.items():
        if k == "value":
            pd.testing.assert_frame_equal(v, loaded_res.value)
        else:
            assert v == getattr(loaded_res, k)


def test_save_load_nested_dataclass(tmp_path: Path):
    @dataclass
    class NestedDataclass:
        a: int
        b: list[str]

    res = ExperimentResult(
        config=ExperimentConfig(
            name="test",
            seed_id=0,
        ),
        execution_time=1.0,
        value=NestedDataclass(a=1, b=["a", "b", "c"]),
    )
    task_id = TaskID(
        action="nest_test_action",
        model="nest_test_model",
        dataset="nest_test_ds",
    )
    set_artifacts_dir(tmp_path / "artifacts")

    save_result(task_id=task_id, result=res)
    loaded_res = load_result(
        config_type=ExperimentConfig,
        value_type=NestedDataclass,
        task_id=task_id,
    )
    assert res.__repr__() == loaded_res.__repr__()


def test_save_load_nested_dataframes(tmp_path: Path):
    @dataclass
    class NestedResult:
        foo: pd.DataFrame
        bar_arr: list[pd.DataFrame]

    res = ExperimentResult(
        config=ExperimentConfig(
            name="test",
            seed_id=0,
        ),
        execution_time=1.0,
        value=NestedResult(
            foo=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            bar_arr=[
                pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
                pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]}),
            ],
        ),
    )
    task_id = TaskID(
        action="nest_df_test_action",
        model="nest_df_test_model",
        dataset="nest_df_test_ds",
    )
    artifacts_dir = tmp_path / "artifacts"
    set_artifacts_dir(artifacts_dir)

    save_result(task_id=task_id, result=res)
    loaded_res = load_result(
        config_type=ExperimentConfig,
        value_type=NestedResult,
        task_id=task_id,
    )
    assert res.__repr__() == loaded_res.__repr__()

    foo_df = pd.read_parquet(
        artifacts_dir
        / "nest_df_test_action"
        / "nest_df_test_model"
        / "nest_df_test_ds"
        / "result_data"
        / "value--foo.df.parquet"
    )
    pd.testing.assert_frame_equal(res.value.foo, foo_df)
    for i, bar_df in enumerate(res.value.bar_arr):
        loaded_bar_df = pd.read_parquet(
            artifacts_dir
            / "nest_df_test_action"
            / "nest_df_test_model"
            / "nest_df_test_ds"
            / "result_data"
            / f"value--bar_arr--{i}.df.parquet"
        )
        pd.testing.assert_frame_equal(bar_df, loaded_bar_df)


def test_save_load_nested_numpy_arrays(tmp_path: Path):
    @dataclass
    class NestedResult:
        foo: np.ndarray
        bar_arr: list[np.ndarray]

    res = ExperimentResult(
        config=ExperimentConfig(
            name="test",
            seed_id=0,
        ),
        execution_time=1.0,
        value=NestedResult(
            foo=np.array([[1, 2, 3], [4, 5, 6]]),
            bar_arr=[
                np.array([[1, 2, 3], [4, 5, 6]]),
                np.array([[7, 8, 9], [10, 11, 12]]),
            ],
        ),
    )
    task_id = TaskID(
        action="nest_array_test_action",
        model="nest_array_test_model",
        dataset="nest_array_test_ds",
    )
    artifacts_dir = tmp_path / "artifacts"
    set_artifacts_dir(artifacts_dir)

    save_result(task_id=task_id, result=res)
    loaded_res = load_result(
        config_type=ExperimentConfig,
        value_type=NestedResult,
        task_id=task_id,
    )
    assert res.__repr__() == loaded_res.__repr__()

    foo_arr = np.load(
        artifacts_dir
        / "nest_array_test_action"
        / "nest_array_test_model"
        / "nest_array_test_ds"
        / "result_data"
        / "value--foo.arr.npy",
        allow_pickle=True,
    )
    np.testing.assert_array_equal(res.value.foo, foo_arr)
    for i, bar_arr in enumerate(res.value.bar_arr):
        loaded_bar_arr = np.load(
            artifacts_dir
            / "nest_array_test_action"
            / "nest_array_test_model"
            / "nest_array_test_ds"
            / "result_data"
            / f"value--bar_arr--{i}.arr.npy",
            allow_pickle=True,
        )
        np.testing.assert_array_equal(bar_arr, loaded_bar_arr)


def test_save_load_paths(tmp_path: Path):
    @dataclass
    class NestedResult:
        foo: int
        bar: Path

    target_path = tmp_path / "foo.txt"
    res = ExperimentResult(
        config=ExperimentConfig(
            name="test",
            seed_id=0,
        ),
        execution_time=1.0,
        value=NestedResult(
            foo=1,
            bar=target_path,
        ),
    )
    task_id = TaskID(
        action="nest_df_test_action",
        model="nest_df_test_model",
        dataset="nest_df_test_ds",
    )
    artifacts_dir = tmp_path / "artifacts"
    set_artifacts_dir(artifacts_dir)

    save_result(task_id=task_id, result=res)
    loaded_res = load_result(
        config_type=ExperimentConfig,
        value_type=NestedResult,
        task_id=task_id,
    )
    assert res.__repr__() == loaded_res.__repr__()
    assert res.value.foo == loaded_res.value.foo


def test_nosave_result(tmp_path: Path):
    @dataclass
    class NestedResult:
        a: int
        omit: NoSave[str]

    res = ExperimentResult(
        config=ExperimentConfig(
            name="test",
            seed_id=0,
        ),
        execution_time=1.0,
        value=NestedResult(
            a=1,
            omit=NoSaveValue("foo"),
        ),
    )
    task_id = TaskID(action="nosave_test_action")
    artifacts_dir = tmp_path / "artifacts"
    set_artifacts_dir(artifacts_dir)

    save_result(task_id=task_id, result=res)
    loaded_res = load_result(
        config_type=ExperimentConfig,
        value_type=NestedResult,
        task_id=task_id,
    )
    assert res.value.a == loaded_res.value.a
    assert loaded_res.value.omit is None
