from __future__ import annotations

from multiprocessing import Queue

import pytest

from pipeline.data import BaseData, DataCollection, ExceptionCloseData
from pipeline.producer import Producer, interruptible
from tests.pipeline.test_data import TData, TException
from util.logging import logger_manager


def test_interruptible() -> None:
    output_queue: 'Queue[DataCollection]' = Queue()

    def generate_data(output_queue: 'Queue[DataCollection]', number: int) -> None:
        output_queue.put(DataCollection().add(TData(number)))
        raise TException()

    interruptible(generate_data, output_queue, 1)

    assert output_queue.qsize() == 2
    assert output_queue.get().has(TData)
    assert output_queue.get().has(ExceptionCloseData)


class MessageData(BaseData):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message


def queue_response(queue: 'Queue[str]', message: str) -> None:
    queue.put(DataCollection().add(MessageData(message)))
    queue.cancel_join_thread()


@pytest.mark.parametrize('handle_logs', [False, True])
def test_producer(handle_logs: bool) -> None:
    if handle_logs:
        logger_manager.start()

    queue: 'Queue[DataCollection]' = Queue()
    producer = Producer(queue, 'message')
    producer.base_start(queue_response, handle_logs=handle_logs)

    message_data = queue.get()
    assert message_data.get(MessageData).message == 'message'

    producer.stop()
    close = queue.get()
    assert isinstance(close, DataCollection)
    assert close.is_closed()

    if handle_logs:
        logger_manager.close()
