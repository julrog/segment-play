import logging
import time
from multiprocessing import Process, Queue
from typing import List, Optional

import cv2
import pytest

from frame.camera import CaptureSettings
from frame.clean import CleanFrameProducer
from frame.producer import FrameData, VideoCaptureProducer
from frame.shared import FramePool, create_frame_pool
from frame.show import WINDOW_NAME, WindowProducer, produce_window
from pipeline.data import CloseData, DataCollection
from pipeline.manager import clear_queue
from util.image import create_black_image


@pytest.mark.parametrize('use_output', [True, False])
def test_produce_window(use_output: bool) -> None:
    input_queue: 'Queue[DataCollection]' = Queue()
    output_queue: Optional['Queue[DataCollection]'] = Queue(
    ) if use_output else None
    key_queue: 'Queue[int]' = Queue()
    frame_pool: FramePool = FramePool(
        create_black_image((100, 100, 3)), 2)
    frame_pools = {FrameData: frame_pool}
    frames: List[FrameData] = [FrameData(create_black_image(
        (100, 100, 3)), frame_pool) for _ in range(2)]

    for frame in frames:
        input_queue.put(DataCollection().add(frame))
    input_queue.put(DataCollection().add(CloseData()))

    produce_window(input_queue, output_queue,
                   key_queue, {FrameData: frame_pool})

    assert input_queue.empty()
    assert not frame_pool.has_free_slots()

    clear_queue(input_queue, frame_pools)
    clear_queue(key_queue)
    if output_queue:
        clear_queue(output_queue, frame_pools)
    else:
        for frame in frames:
            frame_pool.free_frame(frame.frame)


def fill_queue_delay(
    frames: List[FrameData],
    input_queue: 'Queue[DataCollection]'
) -> None:
    time.sleep(0.2)
    for frame in frames:
        input_queue.put(DataCollection().add(frame))
    input_queue.put(DataCollection().add(CloseData()))
    input_queue.cancel_join_thread()


def test_produce_window_delayed_input() -> None:
    input_queue: 'Queue[DataCollection]' = Queue()
    output_queue: 'Queue[DataCollection]' = Queue()
    key_queue: 'Queue[int]' = Queue()
    frame_pool: FramePool = FramePool(
        create_black_image((100, 100, 3)), 2)
    frame_pools = {FrameData: frame_pool}
    frames: List[FrameData] = [FrameData(create_black_image(
        (100, 100, 3)), frame_pool) for _ in range(2)]

    fill_process = Process(target=fill_queue_delay,
                           args=(frames, input_queue))
    fill_process.start()

    produce_window(input_queue, output_queue,
                   key_queue, {FrameData: frame_pool})

    fill_process.join()

    assert input_queue.empty()
    assert not frame_pool.has_free_slots()

    clear_queue(input_queue, frame_pools)
    clear_queue(key_queue)
    clear_queue(output_queue, frame_pools)


def test_window_producer(
    sample_capture_settings: CaptureSettings,
) -> None:
    frame_pool = create_frame_pool(40, sample_capture_settings)
    frame_pools = {FrameData: frame_pool}
    frame_queue: 'Queue[DataCollection]' = Queue()
    clean_queue: 'Queue[DataCollection]' = Queue()
    key_queue: 'Queue[int]' = Queue()

    frame_producer = VideoCaptureProducer(
        frame_queue, sample_capture_settings, frame_pool)
    frame_producer.start()

    clean_producer = CleanFrameProducer(
        frame_queue,
        {FrameData: frame_pool},
        cleanup_delay=0.01,
        limit=10
    )
    clean_producer.start()

    window_producer = WindowProducer(
        frame_queue,
        clean_queue,
        key_queue,
        {FrameData: frame_pool}
    )
    window_producer.start()

    found_window = False
    while True:
        visibility = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
        if visibility < 1:
            found_window = True
            break

        time.sleep(0.01)
    assert found_window

    frame_producer.stop()
    window_producer.join()
    clean_producer.join()
    clear_queue(frame_queue, frame_pools)
    clear_queue(clean_queue, frame_pools)
    clear_queue(key_queue)


def window_producer_logs_fill_queue(
    input_queue: 'Queue[DataCollection]'
) -> None:
    input_queue.put(DataCollection().add(
        FrameData(create_black_image((100, 100, 3)))))
    input_queue.put(DataCollection().add(
        FrameData(create_black_image((100, 100, 3)))))
    input_queue.put(DataCollection().add(
        FrameData(create_black_image((100, 100, 3)))))
    input_queue.put(DataCollection().add(
        FrameData(create_black_image((100, 100, 3)))))
    input_queue.put(DataCollection().add(CloseData()))


def test_window_producer_logs(caplog: pytest.LogCaptureFixture) -> None:
    input_queue: 'Queue[DataCollection]' = Queue()
    key_queue: 'Queue[int]' = Queue()

    input_queue.put(DataCollection().add(
        FrameData(create_black_image((100, 100, 3)))))
    input_queue.put(DataCollection().add(
        FrameData(create_black_image((100, 100, 3)))))
    input_queue.put(DataCollection().add(
        FrameData(create_black_image((100, 100, 3)))))
    input_queue.put(DataCollection().add(
        FrameData(create_black_image((100, 100, 3)))))
    input_queue.put(DataCollection().add(CloseData()))

    with caplog.at_level(logging.INFO):
        produce_window(input_queue, None, key_queue, {
                       FrameData: None}, FrameData, 2)

        assert input_queue.empty()

        clear_queue(input_queue)
        clear_queue(key_queue)

        log_tuples = caplog.record_tuples
        assert len(log_tuples) == 1
        for log_tuple in log_tuples:
            assert log_tuple[0] == 'root'
            assert log_tuple[1] == logging.INFO
            assert log_tuple[2].startswith(f'Window-{WINDOW_NAME}-FPS:')
