from __future__ import annotations

import queue
import time
from multiprocessing import Queue
from typing import Dict, Generator, List, Optional, Type, TypeVar

MISSING_DATA_MESSAGE = 'Missing data in pipeline package!'


class BaseData:
    pass


T = TypeVar('T')


class DataCollection:
    def __init__(
        self,
        data: Optional[Dict[type, BaseData]] = None,
        timestamp: Optional[float] = None
    ) -> None:
        self.data = data if data is not None else {}
        if timestamp:
            self.timestamp = timestamp
        else:
            self.timestamp = time.time()

    def add(self, data: BaseData) -> DataCollection:
        self.data[type(data)] = data
        return self

    def has(self, data_type: Type) -> bool:
        return data_type in self.data

    def get(self, data_type: Type[T]) -> T:
        return self.data.get(data_type)  # type: ignore

    def is_closed(self) -> bool:
        return CloseData in self.data or ExceptionCloseData in self.data


class CloseData(BaseData):
    def __init__(self) -> None:
        super().__init__()


class ExceptionCloseData(CloseData):
    def __init__(self, exception: Exception) -> None:
        super().__init__()
        self.exception = exception


def pipeline_data_generator(
    input_queue: Queue[DataCollection],
    output_queue: Queue[DataCollection],
    expected_data: List[Type],
    timeout: float = 10.0
) -> Generator[DataCollection, None, None]:
    try:
        empty: bool = False
        empty_time: float = time.time()
        while True:
            try:
                data = input_queue.get(timeout=0.01)
                empty = False
                if data.is_closed():
                    output_queue.put(data)
                    break
                assert all(data.has(ed)
                           for ed in expected_data), MISSING_DATA_MESSAGE
                yield data
            except queue.Empty:
                if not empty:
                    empty_time = time.time()
                    empty = True
                if empty_time + timeout < time.time():
                    raise Exception('Pipeline data generator timeout!')
    except KeyboardInterrupt:  # pragma: no cover
        pass
    except Exception as e:
        output_queue.put(DataCollection().add(ExceptionCloseData(e)))


def clear_queue(clear_queue: Queue) -> None:
    try:
        while True:
            clear_queue.get_nowait()
    except queue.Empty:
        pass
    clear_queue.close()
