import functools
import logging
import logging.handlers
import sys
from multiprocessing import Process, Queue
from typing import Any, Callable, Optional


def log_listener_configurer() -> None:
    root = logging.getLogger()
    h = logging.StreamHandler(sys.stdout)
    f = logging.Formatter(
        '%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    h.setFormatter(f)
    root.addHandler(h)


def log_listener_process(queue: 'Queue[Any]', configurer: Callable) -> None:
    configurer()
    while True:
        try:
            record = queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:  # pragma: no cover
            import traceback
            print('Log problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


class ProcessLogger:
    def __init__(self, queue: 'Queue[Any]') -> None:
        self.queue = queue

    def configure(self) -> None:
        h = logging.handlers.QueueHandler(self.queue)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)

    def close(self) -> None:
        self.queue.cancel_join_thread()


class LoggerManager:
    def __init__(self) -> None:
        self.queue: 'Queue[Any]' = Queue(-1)
        self.listener: Optional[Process] = None

    def start(self) -> None:
        if self.listener is None:
            self.listener = Process(
                target=log_listener_process,
                args=(self.queue, log_listener_configurer)
            )
            self.listener.start()

    def create_logger(self) -> ProcessLogger:
        self.start()
        return ProcessLogger(self.queue)

    def close(self) -> None:
        if self.listener is not None:
            self.queue.put(None)
            self.listener.join()
            self.listener = None

    def __del__(self) -> None:
        self.close()


class logging_process:
    def __init__(self, func: Callable[..., None]) -> None:
        self.func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        logger: ProcessLogger = kwargs.pop('logger', None)
        if logger:
            logger.configure()
        self.func(*args, **kwargs)
        if logger:
            logger.close()


logger_manager: LoggerManager = LoggerManager()
