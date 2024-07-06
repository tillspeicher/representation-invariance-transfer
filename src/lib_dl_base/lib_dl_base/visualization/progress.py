import os
from typing import Iterable, TypeVar

from tqdm import tqdm


def should_show_progressbar() -> bool:
    """Returns whether progress bars should be shown."""
    return os.environ.get("DISABLE_PROGRESSBAR", "0") != "1"


T = TypeVar("T", bound=Iterable)


def conditional_tqdm(
    iterable: Iterable[T],
    *args,
    disabled_progress_interval: int | None = None,
    **kwargs,
) -> Iterable[T]:
    """Shows a progress bar only if progress bars aren't disabled."""
    show_progressbar = should_show_progressbar()
    if show_progressbar:
        return tqdm(iterable, *args, **kwargs)
    elif disabled_progress_interval is not None:

        def wrap_iterable(it: Iterable[T], interval: int) -> Iterable[T]:
            for i, item in enumerate(it):
                if i % interval == 0 and i > 0:
                    print(f"Processed {i} items")
                yield item

        return wrap_iterable(iterable, disabled_progress_interval)
    else:
        return iterable
