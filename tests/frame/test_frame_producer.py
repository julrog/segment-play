import queue
import time
from multiprocessing import Queue, Value
from multiprocessing.sharedctypes import Synchronized
from typing import Optional

import numpy as np
import pytest
from conftest import requires_env

from frame.camera import CaptureSettings
from frame.producer import (FrameData, VideoCaptureProducer, free_output_queue,
                            produce_capture)
from frame.shared import FramePool, create_frame_pool
from pipeline.data import DataCollection, clear_queue


def check_frame_data(
        data: DataCollection,
        capture_settings: Optional[CaptureSettings],
        frame_pool: Optional[FramePool] = None
) -> None:
    use_frame_pool = frame_pool is not None
    assert data.has(FrameData)
    frame_data: FrameData = data.get(FrameData)
    assert isinstance(frame_data, FrameData)
    assert frame_data.using_shared_pool == use_frame_pool
    if use_frame_pool:
        assert isinstance(frame_data.frame, int)
    else:
        assert isinstance(frame_data.frame, np.ndarray)
    frame = frame_data.get_frame(frame_pool)
    assert isinstance(frame, np.ndarray)
    if capture_settings is not None:
        assert frame.shape == (
            capture_settings.height, capture_settings.width, 3)


@pytest.mark.parametrize('repetitions', [1, 2, 5])
@pytest.mark.parametrize('queued_items', [0, 1, 5])
@pytest.mark.parametrize('discard_time', [None, 1.0, 0.0])
def test_free_output_queue(
    repetitions: int,
    queued_items: int,
    discard_time: Optional[float]
) -> None:
    output_queue: Queue[DataCollection] = Queue()
    for _ in range(queued_items):
        output_queue.put(DataCollection())
    time.sleep(0.001)  # delay necessary, otherwise empty is not up-to-date
    reduce_frame_discard_timer = discard_time
    for rep in range(repetitions):
        previous_discard_timer = reduce_frame_discard_timer
        reduce_frame_discard_timer = free_output_queue(
            output_queue, None, reduce_frame_discard_timer)
        if previous_discard_timer is not None:
            assert reduce_frame_discard_timer is not None
            if rep + 1 > queued_items:
                if previous_discard_timer == 0:
                    assert previous_discard_timer == reduce_frame_discard_timer
                else:
                    assert previous_discard_timer > reduce_frame_discard_timer
            else:
                assert previous_discard_timer < reduce_frame_discard_timer
        else:
            assert reduce_frame_discard_timer is None
        # delay necessary, otherwise empty is not up-to-date
        time.sleep(0.001)


@pytest.mark.parametrize('use_frame_pool', [False, True])
def test_produce_capture(
        sample_capture_settings: CaptureSettings,
        use_frame_pool: bool
) -> None:
    frame_pool: Optional[FramePool] = create_frame_pool(
        2, sample_capture_settings) if use_frame_pool else None
    frame_queue: Queue[DataCollection] = Queue()
    stop_condition: Synchronized[int] = Value('i', 0)  # type: ignore

    produce_capture(frame_queue, sample_capture_settings,
                    stop_condition, frame_pool)
    time.sleep(0.02)
    data: DataCollection = frame_queue.get()
    assert data.is_closed()
    assert not data.has(FrameData)
    clear_queue(frame_queue)


@pytest.mark.parametrize('use_frame_pool', [False, True])
def test_producer(
    sample_capture_settings: CaptureSettings,
    use_frame_pool: bool
) -> None:
    frame_pool: Optional[FramePool] = create_frame_pool(
        2, sample_capture_settings) if use_frame_pool else None
    frame_queue: Queue[DataCollection] = Queue()

    frame_producer = VideoCaptureProducer(
        frame_queue, sample_capture_settings, frame_pool)
    frame_producer.start()

    data: Optional[DataCollection] = None
    found_frame = False
    while not found_frame:
        try:
            data = frame_queue.get(timeout=0.01)
            assert not data.is_closed()
            found_frame = True
        except queue.Empty:
            pass

    assert isinstance(data, DataCollection)
    check_frame_data(data, sample_capture_settings, frame_pool)

    frame_producer.stop()
    clear_queue(frame_queue)


@pytest.mark.parametrize('frame_pool_size', [5, 20])
@pytest.mark.parametrize('max_queue_size', [None, 10])
def test_producer_no_frame_skips(
    sample_capture_settings: CaptureSettings,
    sample_video_frame_count: int,
    frame_pool_size: int,
    max_queue_size: Optional[int]
) -> None:
    produced_frames = 0
    expected_frames = sample_video_frame_count
    frame_pool: Optional[FramePool] = create_frame_pool(
        frame_pool_size,
        sample_capture_settings
    ) if frame_pool_size > 0 else None
    frame_queue: Queue[DataCollection] = Queue(
        max_queue_size) if max_queue_size is not None else Queue()

    frame_producer = VideoCaptureProducer(
        frame_queue, sample_capture_settings, frame_pool, skip_frames=False)
    frame_producer.start()

    if frame_pool_size == 0:
        queue_max_condition = max_queue_size
    else:
        if max_queue_size is not None:
            queue_max_condition = min(frame_pool_size, max_queue_size)
        else:
            queue_max_condition = frame_pool_size

    if queue_max_condition is not None:
        while frame_queue.qsize() < queue_max_condition:
            time.sleep(0.01)

    time.sleep(0.1)

    if queue_max_condition is not None:
        current_queue_size = frame_queue.qsize()
        # add a little bit of tolerance to the queue size
        assert queue_max_condition + 1 >= current_queue_size
        assert current_queue_size >= queue_max_condition

    while True:
        try:
            if queue_max_condition is not None:
                # add a little bit of tolerance to the queue size
                assert frame_queue.qsize() <= queue_max_condition + 1
            data = frame_queue.get(timeout=0.01)

            assert isinstance(data, DataCollection)
            if produced_frames < expected_frames:
                assert not data.is_closed()
                assert data.has(FrameData)
                if produced_frames % 100 == 0:
                    check_frame_data(
                        data, sample_capture_settings, frame_pool)
                produced_frames += 1
                if frame_pool is not None:
                    frame_pool.free_frame(data.get(FrameData).frame)
            else:
                assert data.is_closed()
                break
        except queue.Empty:
            pass

    assert produced_frames == expected_frames

    assert frame_queue.empty()
    time.sleep(0.1)
    assert frame_queue.empty()

    frame_producer.stop()
    clear_queue(frame_queue)


def test_producer_no_frame_skips_queue_size_limit_on_close(
    sample_capture_settings: CaptureSettings,
    sample_video_frame_count: int,
) -> None:
    max_queue_size = 10
    produced_frames = 0
    expected_frames = sample_video_frame_count
    frame_pool = create_frame_pool(
        20,
        sample_capture_settings
    )

    frame_queue: Queue[DataCollection] = Queue(max_queue_size)

    frame_producer = VideoCaptureProducer(
        frame_queue, sample_capture_settings, frame_pool, skip_frames=False)
    frame_producer.start()

    waited = False
    while True:
        try:
            # add a little bit of tolerance to the queue size
            assert frame_queue.qsize() <= max_queue_size + 1
            data = frame_queue.get(timeout=0.01)

            assert isinstance(data, DataCollection)
            if produced_frames < expected_frames - (max_queue_size + 1):
                assert not data.is_closed()
                produced_frames += 1
                frame_pool.free_frame(data.get(FrameData).frame)
            elif not waited:
                # wait with full queue to test producer waiting with close
                # data until there is some space in the queue
                time.sleep(0.5)
                assert max_queue_size == frame_queue.qsize()
                waited = True
                produced_frames += 1
                frame_pool.free_frame(data.get(FrameData).frame)
            elif produced_frames < expected_frames:
                assert not data.is_closed()
                produced_frames += 1
                frame_pool.free_frame(data.get(FrameData).frame)
            else:
                assert data.is_closed()
                break

        except queue.Empty:
            pass

    assert produced_frames == expected_frames

    assert frame_queue.empty()
    time.sleep(0.1)
    assert frame_queue.empty()

    frame_producer.stop()
    clear_queue(frame_queue)


def test_producer_early_stop(
    sample_capture_settings: CaptureSettings,
) -> None:
    frame_queue: Queue[DataCollection] = Queue()

    producer = VideoCaptureProducer(
        frame_queue, sample_capture_settings)
    producer.stop()

    clear_queue(frame_queue)


@pytest.mark.parametrize('use_settings', [False, True])
@requires_env('cam_tests')
def test_producer_with_camera(use_settings: bool) -> None:
    default_cam_settings: Optional[CaptureSettings] = CaptureSettings(
    ) if use_settings else None
    frame_queue: Queue[DataCollection] = Queue()

    producer = VideoCaptureProducer(
        frame_queue, default_cam_settings)
    producer.start()

    data: Optional[DataCollection] = None
    found_frame = False
    while not found_frame:
        try:
            data = frame_queue.get(block=True, timeout=0.01)
            assert not data.is_closed()
            found_frame = True
        except queue.Empty:
            pass

    assert isinstance(data, DataCollection)
    check_frame_data(data, default_cam_settings)

    producer.stop()
    clear_queue(frame_queue)
