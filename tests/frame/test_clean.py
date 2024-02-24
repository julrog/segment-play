import time
from multiprocessing import Queue
from typing import List, Tuple

import numpy as np
import pytest

from frame.camera import CaptureSettings
from frame.clean import (CleanFrameProducer, clean_frame, filter_frames_limit,
                         filter_old_frames)
from frame.producer import FrameData, VideoCaptureProducer
from frame.shared import FramePool, create_frame_pool
from pipeline.data import CloseData, DataCollection
from pipeline.manager import clear_queue
from util.image import create_black_image


@pytest.mark.parametrize('cleanup_delay', [-0.2, 0.0, 0.2])
def test_filter_old_frames(cleanup_delay: float) -> None:
    number = 5
    addon = int(10 * cleanup_delay)
    frame_pool: FramePool = FramePool(
        create_black_image((100, 100, 3)), number)
    frame_data: List[FrameData] = []
    cleanup_list: List[Tuple[int, float]] = []
    for i in range(number):
        timing = 0.1 + 0.1 * i
        frame_data.append(
            FrameData(create_black_image((100, 100, 3)), frame_pool))
        cleanup_list.append((frame_data[-1].frame, timing))

    assert frame_pool.free_frames.qsize() == 0

    for i in range(number + int(10 * cleanup_delay)):
        timing = 0.11 + 0.1 * i
        filter_old_frames(timing, cleanup_list,
                          frame_pool, cleanup_delay)
        assert len(cleanup_list) == min(number, addon + number - (i + 1))
        for _, frame_time in cleanup_list:
            assert frame_time + cleanup_delay > timing
        assert frame_pool.free_frames.qsize() == max(0, i + 1 - addon)
    assert frame_pool.is_empty()


@pytest.mark.parametrize('limit', [0, 5, 20])
@pytest.mark.parametrize('frames', [0, 5, 20])
def test_filter_frames_limit(limit: int, frames: int) -> None:
    frame_pool: FramePool = FramePool(
        create_black_image((100, 100, 3)), frames)
    frame_data: List[FrameData] = []
    cleanup_list: List[Tuple[int, float]] = []
    for i in range(frames):
        frame_data.append(
            FrameData(create_black_image((100, 100, 3)), frame_pool))
        cleanup_list.append((frame_data[-1].frame, 0.1 * i * (-1)**i))

    assert frame_pool.free_frames.qsize() == 0

    filter_frames_limit(cleanup_list, frame_pool, limit)
    assert len(cleanup_list) == min(limit, frames)
    assert frame_pool.free_frames.qsize() == max(0, frames - limit)

    filter_frames_limit(cleanup_list, frame_pool, 0)
    assert frame_pool.free_frames.qsize() == frames


def test_clean_frame() -> None:
    input_queue: 'Queue[DataCollection]' = Queue()
    frame_pool: FramePool = FramePool(
        create_black_image((100, 100, 3)), 2)

    input_queue.put(DataCollection().add(
        FrameData(create_black_image((100, 100, 3)), frame_pool)))
    input_queue.put(DataCollection().add(
        FrameData(create_black_image((100, 100, 3)), frame_pool)))
    input_queue.put(DataCollection().add(
        FrameData(create_black_image((100, 100, 3)))))
    input_queue.put(DataCollection().add(CloseData()))

    clean_frame(input_queue, frame_pool)

    assert input_queue.empty()
    assert frame_pool.is_empty()


def test_clean_frame_early_close() -> None:
    input_queue: 'Queue[DataCollection]' = Queue()
    frame_pool: FramePool = FramePool(
        create_black_image((100, 100, 3)), 2)

    input_queue.put(DataCollection().add(
        FrameData(create_black_image((100, 100, 3)), frame_pool)))
    input_queue.put(DataCollection().add(
        FrameData(create_black_image((100, 100, 3)), frame_pool)))
    input_queue.put(DataCollection().add(CloseData()))

    clean_frame(input_queue, frame_pool, cleanup_delay=0.1, limit=1)

    assert input_queue.empty()
    assert frame_pool.is_empty()


def test_clean_frame_producer(
    sample_capture_settings: CaptureSettings,
) -> None:
    frame_pool = create_frame_pool(40, sample_capture_settings)
    frame_queue: 'Queue[DataCollection]' = Queue()

    frame_producer = VideoCaptureProducer(
        frame_queue, sample_capture_settings, frame_pool)
    frame_producer.start()

    clean_producer = CleanFrameProducer(
        frame_queue,
        frame_pool,
        cleanup_delay=0.01,
        limit=10
    )
    clean_producer.start()

    time.sleep(0.2)

    frame_producer.stop()
    clean_producer.join()
    clear_queue(frame_queue, frame_pool)


def test_stop_producer() -> None:
    frame_pool = FramePool(np.zeros([10, 10], dtype=np.uint8), 2)
    in_queue: 'Queue[DataCollection]' = Queue()
    producer = CleanFrameProducer(in_queue, frame_pool)
    producer.start()
    producer.stop()

    assert in_queue.empty()

    clear_queue(in_queue, frame_pool)
