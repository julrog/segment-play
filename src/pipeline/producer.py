from __future__ import annotations

import time
from collections.abc import Callable
from multiprocessing import Process, Queue
from typing import Any, Optional

from pipeline.data import CloseData, DataCollection, ExceptionCloseData
from util.logging import logger_manager, logging_process


class Producer:
    def __init__(
        self,
        input_queue: 'Queue[DataCollection]',
        *args: Any,
        **kwargs: Any
    ) -> None:
        self.process: Optional[Process] = None
        self.input_queue = input_queue
        self.args = args
        self.kwargs = kwargs

    def base_start(self, func: Callable) -> None:
        self.process = Process(target=logging_process(func), args=(
            self.input_queue,
            *self.args,
        ), kwargs=dict(self.kwargs, logger=logger_manager.create_logger()))
        self.process.start()

    def stop(self) -> None:
        self.input_queue.put(DataCollection().add(CloseData()))
        if self.process:
            time.sleep(1)
            self.process.kill()

    def join(self) -> None:
        if self.process:
            self.process.join(1.0)
            if self.process.is_alive():  # pragma: no cover
                self.process.kill()


def interruptible(
    fn: Callable[['Queue[DataCollection]', Any], None],
    output_queue: 'Queue[DataCollection]',
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
