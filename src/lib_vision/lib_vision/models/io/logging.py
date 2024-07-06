import pandas as pd
from lightning.pytorch import loggers as pl_loggers

from lib_dl_base.defs.task_id import TaskID
from lib_dl_base.io.dirs import get_logging_dir, rmdir_recursive


def get_tb_logger(
    description: TaskID,
) -> pl_loggers.TensorBoardLogger:
    logging_dir = get_logging_dir(description, "tb")
    return pl_loggers.TensorBoardLogger(
        str(logging_dir),
        name="",
        flush_secs=10,
        version=0,
    )


def get_csv_logger(
    description: TaskID,
) -> pl_loggers.CSVLogger:
    logging_dir = get_logging_dir(description, "csv")
    return pl_loggers.CSVLogger(
        str(logging_dir),
        name="",
        version=0,
    )


def delete_prev_logs(
    description: TaskID,
) -> None:
    logging_dir = get_logging_dir(description, "csv")
    rmdir_recursive(logging_dir)


def load_csv_log(
    description: TaskID,
) -> pd.DataFrame:
    logging_dir = get_logging_dir(description, "csv")
    log_file = logging_dir / "version_0" / "metrics.csv"
    return pd.read_csv(log_file)
