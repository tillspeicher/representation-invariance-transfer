from dataclasses import dataclass, field

import lightning as L
import torch

from lib_dl_base.defs.task_id import TaskID

from ..io.persistence.models import get_checkpoints_callback


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DeviceConfig:
    accelerator: str = field(default_factory=_default_device)
    devices: list[int] = field(default_factory=lambda: [0])


class Trainer(L.Trainer):
    def __init__(
        self,
        task_id: TaskID,
        devices: DeviceConfig,
        enable_checkpointing: bool = True,
        *args,
        **kwargs,
    ) -> None:
        callbacks = []
        if enable_checkpointing:
            callbacks.append(get_checkpoints_callback(task_id))
        return super().__init__(
            *args,
            accelerator=devices.accelerator,
            devices=1 if devices.accelerator == "cpu" else devices.devices,
            enable_checkpointing=enable_checkpointing,
            # TODO: add callbacks from args and kwargs if they are specified
            callbacks=callbacks,
            **kwargs,
        )
