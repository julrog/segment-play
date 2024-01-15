from __future__ import annotations

from multiprocessing import Queue

from pipeline.data import DataCollection, ExceptionCloseData
from pipeline.producer import interruptible
from tests.pipeline.test_data import TestData, TestException


def test_interruptible() -> None:
    output_queue: Queue[DataCollection] = Queue()

    def generate_data(output_queue: Queue[DataCollection], number: int) -> None:
        output_queue.put(DataCollection().add(TestData(number)))
        raise TestException()

    interruptible(generate_data, output_queue, 1)

    assert output_queue.qsize() == 2
    assert output_queue.get().has(TestData)
    assert output_queue.get().has(ExceptionCloseData)
