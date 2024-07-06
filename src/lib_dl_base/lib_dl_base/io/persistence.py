import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import dacite
import jsonpickle
import numpy as np
import pandas as pd

from lib_dl_base.defs.task_id import TaskID
from lib_dl_base.io.dirs import append_directories, get_artifacts_dir


T = TypeVar("T")


@dataclass
class NoSaveValue(Generic[T]):
    """A container for values that should not be persisted."""

    value: T


NoSave = NoSaveValue[T] | None

NUMPY_SUFFIXES = [".arr", ".npy"]
NUMPY_SUFFIX = "".join(NUMPY_SUFFIXES)
DATAFRAME_SUFFIXES = [".df", ".parquet"]
DATAFRAME_SUFFIX = "".join(DATAFRAME_SUFFIXES)
PICKLE_SUFFIX = ".pkl"


def save(
    obj: Any,
    task_id: TaskID,
    file_name: str,
) -> None:
    """Save result objects to disk in json format. Pandas DataFrames
    and numpy arrays are saved separately as parquet and npy files
    and referenced in the json file.
    """
    results_dir = get_results_dir(task_id, create_if_not_exists=True)
    if isinstance(obj, np.ndarray):
        _save_numpy_array(obj, results_dir / file_name)
        return
    elif isinstance(obj, pd.DataFrame):
        _save_dataframe(obj, results_dir / file_name)
        return
    else:
        # Make sure obj is a dataclass
        if not hasattr(obj, "__dataclass_fields__"):
            raise TypeError("Object must be a dataclass")

    result_data_dir = results_dir / "result_data"
    result_appendix_files = _register_serialization_handlers(result_data_dir)
    serialized_result = cast(
        str, jsonpickle.encode(obj, indent=2, unpicklable=False)
    )
    # Save the dataframes referenced in the result separately and store
    # pointers to them in the json file
    converted_result = _save_result_appendix_data(
        serialized_result,
        result_appendix_files,
        result_data_dir,
    )
    serialized_result = json.dumps(converted_result, indent=2)
    with open(results_dir / f"{file_name}.json", "w") as res_file:
        res_file.write(serialized_result)


def load(
    task_id: TaskID,
    file_name: str,
    obj_type: type,
) -> Any:
    results_dir = get_results_dir(task_id)
    file_path = results_dir / file_name

    if obj_type == np.ndarray:
        return _load_numpy_array(file_path)
    elif obj_type == pd.DataFrame:
        return _load_dataframe(file_path)
    elif file_path.suffix == PICKLE_SUFFIX:
        return _load_pickle_legacy(file_path)
    else:
        return _load_jsonpickle(file_path, obj_type)


def _load_jsonpickle(
    file_path: Path,
    obj_type: type,
) -> Any:
    result_data_dir = file_path.parent / "result_data"
    _register_serialization_handlers(result_data_dir)
    with open(file_path.with_suffix(".json"), "r") as res_file:
        serialized_result = res_file.read()
    parsed_result = cast(dict, jsonpickle.decode(serialized_result))

    return dacite.from_dict(
        obj_type,
        parsed_result,
    )


def _load_pickle_legacy(
    file_path,
) -> Any:
    try:
        with open(file_path.with_suffix(PICKLE_SUFFIX), "rb") as res_file:
            return pickle.load(res_file)
    except ModuleNotFoundError:
        return None


def get_results_dir(
    task_id: TaskID,
    create_if_not_exists: bool = False,
) -> Path:
    # Where to save the results files (metadata, metrics)
    results_dir = append_directories(get_artifacts_dir(), task_id)
    if create_if_not_exists:
        results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


DATAFRAME_OBJ_ID = "pandas.core.frame.DataFrame"
NUMPY_OBJ_ID = "numpy.ndarray"


def _register_serialization_handlers(
    result_data_dir: Path,
) -> list[pd.DataFrame | np.ndarray]:
    result_data = []

    class PandasHandler(jsonpickle.handlers.BaseHandler):
        def flatten(self, df: pd.DataFrame, data: dict) -> dict:
            result_data.append(df)
            return {
                "py/object": DATAFRAME_OBJ_ID,
                "file_idx": len(result_data) - 1,
            }

        def restore(self, data: dict) -> pd.DataFrame:
            file_name = data["file_name"]
            df_path = result_data_dir / file_name
            return _load_dataframe(df_path)

    jsonpickle.handlers.register(pd.DataFrame, PandasHandler, base=True)

    class NumpyHandler(jsonpickle.handlers.BaseHandler):
        def flatten(self, array: np.ndarray, data: dict) -> dict:
            result_data.append(array)
            return {
                "py/object": NUMPY_OBJ_ID,
                "file_idx": len(result_data) - 1,
            }

        def restore(self, data: dict) -> np.ndarray:
            file_name = data["file_name"]
            array_path = result_data_dir / file_name
            return _load_numpy_array(array_path)

    jsonpickle.handlers.register(np.ndarray, NumpyHandler, base=True)

    class PathHandler(jsonpickle.handlers.BaseHandler):
        def flatten(self, path: Path, data: dict) -> dict:
            return {
                "py/object": "pathlib.Path",
                "file_name": str(path),
            }

        def restore(self, data: dict) -> Path:
            file_name = data["file_name"]
            return Path(file_name)

    jsonpickle.handlers.register(Path, PathHandler, base=True)

    class NoSaveHandler(jsonpickle.handlers.BaseHandler):
        def flatten(self, no_save: NoSave, data: dict) -> None:
            return None

        def restore(self, data: dict) -> NoSave | None:
            return None

    jsonpickle.handlers.register(NoSaveValue, NoSaveHandler, base=True)

    return result_data


def _save_result_appendix_data(
    flattened_json: str,
    result_appendix_data: list[pd.DataFrame | np.ndarray],
    result_data_dir: Path,
) -> dict:
    """Save the additional objects contained in the result
    (pd.DataFrames and np.ndarrays) as separate
    parquet and npy files and reference them in the json file.
    """
    result_dict = json.loads(flattened_json)
    if len(result_appendix_data) == 0:
        return result_dict

    if not result_data_dir.exists():
        result_data_dir.mkdir(parents=True, exist_ok=True)
    _iter_save_data(result_dict, result_appendix_data, result_data_dir)
    return result_dict


def _iter_save_data(
    result_obj: dict | list,
    result_data: list[pd.DataFrame | np.ndarray],
    result_data_dir: Path,
    cur_prefix: list[str] = [],
) -> None:
    if isinstance(result_obj, dict):
        if set(result_obj.keys()) == {"py/object", "file_idx"} and result_obj[
            "py/object"
        ] in [DATAFRAME_OBJ_ID, NUMPY_OBJ_ID]:
            res_data_idx = result_obj["file_idx"]
            res_data = result_data[res_data_idx]
            obj_file = "--".join(cur_prefix)
            result_file = result_data_dir / obj_file
            if result_obj["py/object"] == NUMPY_OBJ_ID:
                _save_numpy_array(res_data, result_file)
            else:
                _save_dataframe(res_data, result_file)

            del result_obj["file_idx"]
            result_obj["file_name"] = obj_file
        else:
            for key, value in result_obj.items():
                _iter_save_data(
                    value,
                    result_data,
                    result_data_dir,
                    cur_prefix + [key],
                )
    elif isinstance(result_obj, list):
        for i, item in enumerate(result_obj):
            _iter_save_data(
                item,
                result_data,
                result_data_dir,
                cur_prefix + [str(i)],
            )
    else:
        return


def _save_numpy_array(array: np.ndarray, file: Path) -> None:
    np.save(file=file.with_suffix(".arr.npy"), arr=array)


def _load_numpy_array(file: Path) -> np.ndarray:
    if not file.suffixes == NUMPY_SUFFIXES:
        file = file.with_suffix(NUMPY_SUFFIX)
    return np.load(file=file, allow_pickle=True)


def _save_dataframe(df: pd.DataFrame, file: Path) -> None:
    df.to_parquet(path=file.with_suffix(DATAFRAME_SUFFIX))


def _load_dataframe(file: Path) -> pd.DataFrame:
    if not file.suffixes == DATAFRAME_SUFFIXES:
        file = file.with_suffix(DATAFRAME_SUFFIX)
    return pd.read_parquet(file)


# RerunCondition = Literal["always", "no_prior_res", "never"]
# Res = TypeVar("Res", bound=ExperimentResult)


# def cached_result(
#     func: Callable[..., Iterator[Res]],

# ) -> Callable[..., Optional[Res]]:
#     @functools.wraps(func)
#     def wrapper_func(
#         description: TaskID,
#         rerun_if: RerunCondition,
#         *args,
#         result_name: str = "result",
#         **kwargs,
#     ) -> Optional[Res]:
#         if rerun_if not in {"always", "no_prior_res", "never"}:
#             raise ValueError()

#         result: Optional[Res] = None
#         if rerun_if == "always" or (
#             rerun_if == "no_prior_res"
#             and not _result_exists(description, result_name)
#         ):
#             for result in func(*args, **kwargs):
#                 # result = func(*args, **kwargs)
#                 save_result(result, description, result_name)
#         elif rerun_if != "never":
#             print("Loading cached results")
#             result = load_result(
#                 config_type=,
#                 task_id=description,
#                 result_name=result_name
#             )
#         return result

#     return wrapper_func


# def _result_exists(
#     description: TaskID,
#     result_name: str,
# ) -> bool:
#     results_dir = get_results_dir(description)
#     results_file = results_dir / f"{result_name}.pkl"
#     return results_file.exists()
