from __future__ import annotations

import time
from collections.abc import Callable
from multiprocessing import Process, Queue, Value
from multiprocessing.sharedctypes import Synchronized
from typing import Any, Optional

from frame.shared import FramePool
from pipeline.data import CloseData, DataCollection, ExceptionCloseData


class Producer:
    def __init__(
        self,
        input_queue: 'Queue[DataCollection]',
        output_queue: 'Queue[DataCollection]',
        frame_pool: Optional[FramePool] = None,
        skip_frames: bool = True,
        log_cycles: int = 100,
        *args: Any,
        **kwargs: Any
    ) -> None:
        self.process: Optional[Process] = None
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.frame_pool = frame_pool
        self.skip_frames = skip_frames
        self.log_cycles = log_cycles
        self.ready: Synchronized[int] = Value('i', 0)  # type: ignore
        self.args = args
        self.kwargs = kwargs

    def base_start(self, func: Callable) -> None:
        self.process = Process(target=func, args=(
            self.input_queue,
            self.output_queue,
            self.ready,
            self.frame_pool,
            self.skip_frames,
            self.log_cycles,
            *self.args,
            *self.kwargs,
        ))
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
