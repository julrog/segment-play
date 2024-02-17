from __future__ import annotations

from multiprocessing import Queue

from pipeline.data import DataCollection, ExceptionCloseData
from pipeline.producer import interruptible
from tests.pipeline.test_data import TData, TException


def test_interruptible() -> None:
    output_queue: 'Queue[DataCollection]' = Queue()

    def generate_data(output_queue: 'Queue[DataCollection]', number: int) -> None:
        output_queue.put(DataCollection().add(TData(number)))
        raise TException()

    interruptible(generate_data, output_queue, 1)

    assert output_queue.qsize() == 2
    assert output_queue.get().has(TData)
    assert output_queue.get().has(ExceptionCloseData)
