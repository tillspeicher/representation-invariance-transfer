from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional


@dataclass
class TaskID:
    action: Optional[str] = None
    model: Optional[str] = None
    dataset: Optional[str] = None
    task_prefixes: list[str] = field(default_factory=list)

    @property
    def _prefix(self) -> list[str]:
        return []

    @property
    def path(self) -> Path:
        path = Path(*self._prefix)
        for prefix in self.task_prefixes:
            path /= prefix
        if self.action is not None:
            path /= self.action
        if self.model is not None:
            path /= self.model
        if self.dataset is not None:
            path /= self.dataset
        return path

    @property
    def name(self) -> str:
        name = self._prefix
        for prefix in self.task_prefixes:
            name += [prefix]
        if self.action is not None:
            name.append(self.action)
        if self.model is not None:
            name.append(self.model)
        if self.dataset is not None:
            name.append(self.dataset)
        return str(name)

    def set_action(self, action: str) -> "TaskID":
        return replace(self, action=action)

    def set_model(self, model: str) -> "TaskID":
        return replace(self, model=model)

    def set_dataset(self, dataset: str) -> "TaskID":
        return replace(self, dataset=dataset)

    def add_prefix(self, prefix: str) -> "TaskID":
        return replace(self, task_prefixes=self.task_prefixes + [prefix])
