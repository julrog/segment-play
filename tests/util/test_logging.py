import logging
from multiprocessing import Process, Queue
from typing import Any

import pytest

from util.logging import (LoggerManager, ProcessLogger,
                          log_listener_configurer, log_listener_process,
                          logger_manager, logging_process)


def log_producer() -> None:
    for _ in range(3):
        logging.info('Test log message')


def test_logging(caplog: pytest.LogCaptureFixture) -> None:
    lm = LoggerManager()
    in_process_logger = lm.create_logger()
    in_process_logger.configure()

    with caplog.at_level(logging.INFO):
        logging.info('Test log message')
        lm.close()

        log_tuples = caplog.record_tuples
        assert len(log_tuples) == 1
        for log_tuple in log_tuples:
            assert log_tuple[0] == 'root'
            assert log_tuple[1] == logging.INFO
            assert log_tuple[2] == 'Test log message'


def test_logging_handler() -> None:
    logger_manager.start()
    logger_manager.start()
    process = Process(target=logging_process(
        log_producer), kwargs=dict(logger=logger_manager.create_logger()))
    process.start()
    process.join()
    logger_manager.close()
    logger_manager.close()


def test_logging_in_process(caplog: pytest.LogCaptureFixture) -> None:
    queue: 'Queue[Any]' = Queue(-1)
    process_logger = ProcessLogger(queue)

    process = Process(target=logging_process(
        log_producer), kwargs=dict(logger=process_logger))

    with caplog.at_level(logging.INFO):
        process.start()
        process.join()
        queue.put(None)

        log_listener_process(queue, log_listener_configurer)

        log_tuples = caplog.record_tuples
        assert len(log_tuples) == 3
        for log_tuple in log_tuples:
            assert log_tuple[0] == 'root'
            assert log_tuple[1] == logging.INFO
            assert 'Test log message' in log_tuple[2]


def test_logging_in_process_no_logs(caplog: pytest.LogCaptureFixture) -> None:
    queue: 'Queue[Any]' = Queue(-1)
    process = Process(target=logging_process(log_producer))

    with caplog.at_level(logging.INFO):
        process.start()
        process.join()
        queue.put(None)

        log_listener_process(queue, log_listener_configurer)

        log_tuples = caplog.record_tuples
        assert len(log_tuples) == 0
