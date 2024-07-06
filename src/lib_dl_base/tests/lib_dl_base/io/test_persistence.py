from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib_dl_base.defs.task_id import TaskID
from lib_dl_base.io.dirs import set_artifacts_dir
from lib_dl_base.io.persistence import NoSave, NoSaveValue, load, save


@pytest.fixture
def temp_dir(tmpdir):
    artifacts_dir = Path(tmpdir) / "artifacts"
    set_artifacts_dir(artifacts_dir)
    return artifacts_dir


def test_reject_non_dataclasses(temp_dir):
    task_id = TaskID(
        action="test_action",
        model="test_model",
        dataset="test_dataset",
    )
    obj = {"key1": "value1", "key2": 2}
    file_name = "test_object"

    with pytest.raises(TypeError):
        save(obj, task_id, file_name)


@dataclass
class PlainObject:
    field1: str
    field2: int


def test_save_load_plain_object(temp_dir):
    task_id = TaskID(
        action="test_action",
        model="test_model",
        dataset="test_dataset",
    )
    obj = PlainObject(field1="value1", field2=2)
    file_name = "test_object"

    save(obj, task_id, file_name)
    loaded_obj = load(task_id, file_name, type(obj))

    assert obj == loaded_obj


@dataclass
class ObjectWithNestedPlainObjects:
    field1: str
    field2: list[PlainObject]


def test_save_load_object_with_nested_plain_objects(temp_dir):
    task_id = TaskID(
        action="test_action",
        model="test_model",
        dataset="test_dataset",
    )
    obj = ObjectWithNestedPlainObjects(
        field1="value1",
        field2=[
            PlainObject(field1="value2", field2=3),
            PlainObject(field1="value3", field2=4),
        ],
    )
    file_name = "test_object_with_nested_plain_objects"

    save(obj, task_id, file_name)
    loaded_obj = load(task_id, file_name, type(obj))

    assert obj.field1 == loaded_obj.field1
    assert obj.field2 == loaded_obj.field2


def test_save_load_numpy_array(temp_dir):
    task_id = TaskID(
        action="test_action",
        model="test_model",
        dataset="test_dataset",
    )
    obj = np.array([1, 2, 3, 4, 5])
    file_name = "test_array"

    save(obj, task_id, file_name)
    loaded_obj = load(task_id, file_name, type(obj))

    assert np.array_equal(obj, loaded_obj)


@dataclass
class ObjectWithNumpyArray:
    field1: str
    field2: np.ndarray


def test_save_load_object_with_numpy_array(temp_dir):
    task_id = TaskID(
        action="test_action",
        model="test_model",
        dataset="test_dataset",
    )
    obj = ObjectWithNumpyArray(field1="value1", field2=np.array([1, 2, 3]))
    file_name = "test_object_with_numpy_array"

    save(obj, task_id, file_name)
    loaded_obj = load(task_id, file_name, type(obj))

    assert obj.field1 == loaded_obj.field1
    assert np.array_equal(obj.field2, loaded_obj.field2)


@dataclass
class ObjectWithNestedNumpyArrays:
    field1: str
    field2: list[np.ndarray]


def test_save_load_object_with_nested_numpy_arrays(temp_dir):
    task_id = TaskID(
        action="test_action",
        model="test_model",
        dataset="test_dataset",
    )
    obj = ObjectWithNestedNumpyArrays(
        field1="value1", field2=[np.array([1, 2]), np.array([3, 4, 5])]
    )
    file_name = "test_object_with_nested_numpy_arrays"

    save(obj, task_id, file_name)
    loaded_obj = load(task_id, file_name, type(obj))

    assert obj.field1 == loaded_obj.field1
    assert all(
        np.array_equal(a, b) for a, b in zip(obj.field2, loaded_obj.field2)
    )


def test_save_load_pandas_dataframe(temp_dir):
    task_id = TaskID(
        action="test_action",
        model="test_model",
        dataset="test_dataset",
    )
    obj = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    file_name = "test_dataframe"

    save(obj, task_id, file_name)
    loaded_obj = load(task_id, file_name, type(obj))

    pd.testing.assert_frame_equal(obj, loaded_obj)


@dataclass
class ObjectWithPandasDataframe:
    field1: str
    field2: pd.DataFrame


def test_save_load_object_with_pandas_dataframe(temp_dir):
    task_id = TaskID(
        action="test_action",
        model="test_model",
        dataset="test_dataset",
    )
    obj = ObjectWithPandasDataframe(
        field1="value1", field2=pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    )
    file_name = "test_object_with_pandas_dataframe"

    save(obj, task_id, file_name)
    loaded_obj = load(task_id, file_name, type(obj))

    assert obj.field1 == loaded_obj.field1
    pd.testing.assert_frame_equal(obj.field2, loaded_obj.field2)


@dataclass
class ObjectWithNestedPandasDataframes:
    field1: str
    field2: list[pd.DataFrame]


def test_save_load_object_with_nested_pandas_dataframes(temp_dir):
    task_id = TaskID(
        action="test_action",
        model="test_model",
        dataset="test_dataset",
    )
    obj = ObjectWithNestedPandasDataframes(
        field1="value1",
        field2=[
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            pd.DataFrame({"C": [5, 6], "D": [7, 8]}),
        ],
    )
    file_name = "test_object_with_nested_pandas_dataframes"

    save(obj, task_id, file_name)
    loaded_obj = load(task_id, file_name, type(obj))

    assert obj.field1 == loaded_obj.field1
    assert all(
        pd.testing.assert_frame_equal(a, b) is None
        for a, b in zip(obj.field2, loaded_obj.field2)
    )


@dataclass
class ObjectWithNoSaveValue:
    field1: str
    field2: NoSave[int]


def test_save_load_nosave_value(temp_dir):
    task_id = TaskID(
        action="test_action",
        model="test_model",
        dataset="test_dataset",
    )
    obj = ObjectWithNoSaveValue(field1="value1", field2=NoSaveValue(42))
    file_name = "test_nosave"

    save(obj, task_id, file_name)
    loaded_obj = load(task_id, file_name, type(obj))

    assert obj.field1 == loaded_obj.field1
    assert loaded_obj.field2 is None
