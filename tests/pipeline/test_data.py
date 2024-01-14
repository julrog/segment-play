from __future__ import annotations

import time
from multiprocessing import Process, Queue

from pipeline.data import (MISSING_DATA_MESSAGE, BaseData, CloseData,
                           DataCollection, ExceptionCloseData,
                           pipeline_data_generator)


class TestData(BaseData):
    def __init__(self, value: int) -> None:
        super().__init__()
        self.value = value


def test_data_collection() -> None:
    current_time = time.time()
    data = DataCollection(timestamp=current_time)
    assert current_time == data.timestamp
    time.sleep(0.01)
    data = DataCollection()
    assert current_time < data.timestamp

    assert not data.has(BaseData)
    base_data = BaseData()
    data.add(base_data)
    assert data.has(BaseData)

    assert not data.has(TestData)
    test_data = TestData(1)
    data.add(test_data)
    assert data.has(TestData)
    assert data.get(TestData).value == 1


def test_close_data() -> None:
    data = DataCollection()
    assert not data.is_closed()
    data.add(CloseData())
    assert data.is_closed()


def test_exception_data() -> None:
    class TestException(Exception):
        message = 'Test'

    data = DataCollection()
    assert not data.is_closed()
    data.add(ExceptionCloseData(TestException()))
    assert data.is_closed()

    assert data.get(ExceptionCloseData).exception.message == 'Test'


def fill_queue(input_queue: Queue[DataCollection], count: int, missing: bool) -> None:
    for i in range(count):
        input_queue.put(DataCollection().add(TestData(i + 1)))
    if missing:
        input_queue.put(DataCollection())
    input_queue.put(DataCollection().add(CloseData()))


def test_pipeline_data_generator() -> None:
    input_queue: Queue[DataCollection] = Queue(4)
    output_queue: Queue[DataCollection] = Queue(4)

    fill_process = Process(target=fill_queue, args=(input_queue, 3, False))
    fill_process.start()

    count = 1
    for data in pipeline_data_generator(input_queue, output_queue, [TestData]):
        if count < 4:
            assert not data.is_closed()
            assert data.get(TestData).value == count
        else:
            assert data.is_closed()
        count += 1
    assert count == 4

    fill_process.join()


def test_pipeline_data_generator_exception() -> None:
    input_queue: Queue[DataCollection] = Queue(4)
    output_queue: Queue[DataCollection] = Queue(4)

    fill_process = Process(target=fill_queue, args=(input_queue, 2, True))
    fill_process.start()

    count = 1
    for data in pipeline_data_generator(input_queue, output_queue, [TestData]):
        if count < 3:
            assert not data.is_closed()
            assert data.get(TestData).value == count
        elif count == 3:
            assert not data.is_closed()
        else:
            assert data.is_closed()
        count += 1
    assert count == 3

    assert output_queue.qsize() == 1
    outdata = output_queue.get()
    assert outdata.is_closed()
    assert outdata.has(ExceptionCloseData)
    assert str(outdata.get(
        ExceptionCloseData).exception) == MISSING_DATA_MESSAGE

    fill_process.join()
