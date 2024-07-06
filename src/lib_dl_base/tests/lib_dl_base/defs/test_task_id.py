from lib_dl_base.defs.task_id import TaskID


def test_task_id() -> None:
    task_id = TaskID(
        action="test_action",
        model="test_model",
        dataset="test_dataset",
    )
    assert task_id.name == str([task_id.action, task_id.model, task_id.dataset])
