from __future__ import annotations

from collections.abc import Callable
from multiprocessing import Queue
from typing import Any

from pipeline.data import DataCollection, ExceptionCloseData


def interruptible(
    fn: Callable[[Queue[DataCollection], Any], None],
    output_queue: Queue[DataCollection],
    *args: Any,
    **kwargs: Any
) -> None:
    assert output_queue is not None
    try:
        fn(output_queue, *args, **kwargs)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    except Exception as e:
        output_queue.put(DataCollection().add(ExceptionCloseData(e)))
