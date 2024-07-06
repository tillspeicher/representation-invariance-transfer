from dataclasses import dataclass, field, make_dataclass
from typing import Generic, TypeVar, cast

from lib_dl_base.defs.task_id import TaskID
from lib_dl_base.io.persistence import NoSave, NoSaveValue, load, save


DEFAULT_RESULT_NAME = "result"


@dataclass
class ExperimentConfig:
    name: str
    seed_id: int
    group: str | None = field(default=None, kw_only=True)


# Config
C = TypeVar("C", bound=ExperimentConfig)
# Result
R = TypeVar("R")


@dataclass
class ExperimentResult(Generic[C, R]):
    config: C
    value: R
    execution_time: float = 0.0


T = TypeVar("T")


def save_result(
    result: ExperimentResult,
    task_id: TaskID,
    result_name: str = DEFAULT_RESULT_NAME,
) -> None:
    save(result, task_id, result_name)


def load_result(
    config_type: type[C],
    value_type: type[R],
    task_id: TaskID,
    result_name: str = DEFAULT_RESULT_NAME,
) -> ExperimentResult[C, R]:
    custom_exp_res_type = make_dataclass(
        "ExperimentResult",
        [("config", config_type), ("value", value_type)],
        bases=(ExperimentResult,),
    )
    return cast(
        ExperimentResult[C, R],
        load(
            task_id,
            result_name,
            custom_exp_res_type,
        ),
    )
