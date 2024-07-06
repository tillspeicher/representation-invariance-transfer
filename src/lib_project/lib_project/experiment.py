import os
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    Iterable,
    ParamSpec,
    cast,
)

from omegaconf import DictConfig, OmegaConf, SCMode

from lib_dl_base.defs.task_id import TaskID

from .io.experiment_results import (
    C,
    ExperimentConfig,
    ExperimentResult,
    NoSave,
    NoSaveValue,
    R,
    load_result,
    save_result,
)


LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


@dataclass
class ExperimentHandle(Generic[C]):
    id: str
    # config: type[C]
    create_configs: Callable[[], list[C]]
    set_seed: Callable[[C, int], C]
    experiment: Callable[[C], Any]


@dataclass
class ExperimentID(TaskID):
    # TODO: the default values are only there to make inheritance work
    experiment_name: str = ""
    config_name: str | list[str] = ""
    seed_id: int = -1

    @property
    def _prefix(self) -> list[str]:
        config_name_items = (
            [self.config_name]
            if isinstance(self.config_name, str)
            else self.config_name
        )
        return [self.experiment_name, *config_name_items, f"sid_{self.seed_id}"]


P = ParamSpec("P")


def experiment(
    exp_name: str,
) -> Callable[
    [Callable[Concatenate[C, ExperimentID, P], R]],
    Callable[Concatenate[C, P], ExperimentResult[C, R]],
]:
    def experiment_decorator(
        func: Callable[Concatenate[C, ExperimentID, P], R]
    ) -> Callable[Concatenate[C, P], ExperimentResult[C, R]]:
        # Wrap the single-return function to return an iterable
        def iterable_wrapper(
            config: C,
            exp_id: ExperimentID,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Iterable[R]:
            yield func(config, exp_id, *args, **kwargs)

        decorated_function = iterative_experiment(exp_name)(iterable_wrapper)

        def single_result_wrapper(
            config: C,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> ExperimentResult[C, R]:
            # Unpack the single result from the iterable
            return next(iter(decorated_function(config, *args, **kwargs)))

        return single_result_wrapper

    return experiment_decorator


def iterative_experiment(
    exp_name: str,
) -> Callable[
    [Callable[Concatenate[C, ExperimentID, P], Iterable[R]]],
    Callable[Concatenate[C, P], Iterable[ExperimentResult[C, R]]],
]:
    def experiment_decorator(
        func: Callable[Concatenate[C, ExperimentID, P], Iterable[R]]
    ) -> Callable[Concatenate[C, P], Iterable[ExperimentResult[C, R]]]:
        def experiment_wrapper(
            config: C,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Iterable[ExperimentResult[C, R]]:
            task_description = ExperimentID(
                experiment_name=exp_name,
                config_name=(
                    config.name
                    if config.group is None
                    else [config.group, config.name]
                ),
                seed_id=config.seed_id,
            )
            if isinstance(config, DictConfig):
                # OmegaConf converts the dataclass configs to its own classes
                # which can be annoying. Here, we convert them back to the
                # original config classes.
                dataclass_config = cast(
                    C,
                    OmegaConf.to_container(
                        config,
                        structured_config_mode=SCMode.INSTANTIATE,
                    ),
                )
            else:
                dataclass_config = config
            print(
                "=============================\n"
                f"Running experiment {task_description.name}\n"
                "=============================\n"
            )

            time_start = time.time()

            def get_runtime() -> float:
                return time.time() - time_start

            def produce_result(value: R) -> ExperimentResult[C, R]:
                result = ExperimentResult(
                    dataclass_config,
                    value,
                    execution_time=get_runtime(),
                )
                if LOCAL_RANK == 0:
                    # Only save the result in one process, to avoid
                    # race conditions
                    save_result(result, task_description)
                return result

            value_iter = func(
                dataclass_config, task_description, *args, **kwargs
            )
            for value in value_iter:
                yield produce_result(value)
            print(f"=== Experiment finished in {get_runtime():.1f}s! ===")

        return experiment_wrapper

    return experiment_decorator


def load_results(
    experiment_name: str,
    config_name: str | list[str],
    seed_ids: list[int],
    config_type: type[C],
    value_type: type[R],
) -> list[ExperimentResult[C, R]]:
    results = []
    for seed_id in seed_ids:
        try:
            result = load_result(
                config_type=config_type,
                value_type=value_type,
                task_id=ExperimentID(
                    experiment_name=experiment_name,
                    config_name=config_name,
                    seed_id=seed_id,
                ),
            )
            results.append(result)
        except FileNotFoundError as e:
            print(
                f"Could not find result for "
                f"{config_name}, seed {seed_id}: " + e.filename
            )
    return results
