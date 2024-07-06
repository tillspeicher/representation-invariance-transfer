from typing import TypeVar


T = TypeVar("T")


def not_none(value: T | None) -> T:
    assert value is not None
    return value
