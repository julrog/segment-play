import logging
import queue
from multiprocessing import Queue
from typing import Optional

import numpy as np
import pytest

from frame.camera import CaptureSettings
from frame.producer import FrameData, VideoCaptureProducer
from frame.shared import FramePool, create_frame_pool
from pipeline.data import (CloseData, DataCollection, ExceptionCloseData,
                           clear_queue)
from tests.frame.test_frame_producer import check_frame_data
from tracking.producer import TrackingData, TrackProducer, produce_tracking
from util.image import create_black_image


def check_tracking_data(
        data: DataCollection,
        capture_settings: Optional[CaptureSettings],
        frame_pool: Optional[FramePool] = None
) -> None:
    check_frame_data(data, capture_settings, frame_pool)
    assert data.has(TrackingData)
    tracking_data: TrackingData = data.get(TrackingData)
    assert isinstance(tracking_data, TrackingData)
    for id in range(len(tracking_data.targets)):
        assert tracking_data.targets[id].shape == (9,)

        box = tracking_data.get_box(id)
        assert isinstance(box, np.ndarray)
        assert box.shape == (4,)

        padded_box = tracking_data.get_padded_box(id)
        assert isinstance(padded_box, np.ndarray)
        assert padded_box.shape == (4,)

        tid = tracking_data.get_tracking_id(id)
        assert isinstance(tid, int)


def test_tracking_data() -> None:
    tracking_data = TrackingData(
        np.array([
            [0, 0, 100, 100, 1, 20, 20, 80, 80],
            [0, 0, 100, 100, 2, 20, 20, 80, 80]
        ]))

    for id in [0, 1]:
        box = tracking_data.get_box(id)
        assert (box == np.array([0, 0, 100, 100])).all()
        assert box.shape == (4,)

        padded_box = tracking_data.get_padded_box(id)
        assert (padded_box == np.array([20, 20, 80, 80])).all()
        assert padded_box.shape == (4,)

        trackin_id = tracking_data.get_tracking_id(id)
        assert trackin_id == id + 1


@pytest.mark.parametrize('use_frame_pool', [False, True])
def test_produce_tracking(use_frame_pool: bool) -> None:
    input_queue: 'Queue[DataCollection]' = Queue()
    output_queue: 'Queue[DataCollection]' = Queue()
    frame_pool: Optional[FramePool] = FramePool(
        create_black_image((100, 100, 3)), 2) if use_frame_pool else None

    input_queue.put(DataCollection().add(
        FrameData(np.zeros((100, 100, 3), dtype=np.uint8))))
    input_queue.put(DataCollection().add(
        FrameData(np.zeros((100, 100, 3), dtype=np.uint8))))
    input_queue.put(DataCollection().add(CloseData()))

    produce_tracking(input_queue, output_queue,
                     skip_frames=False, frame_pool=frame_pool)

    assert output_queue.qsize() == 3
    assert isinstance(output_queue.get(), DataCollection)
    assert isinstance(output_queue.get(), DataCollection)
    close_data = output_queue.get()
    assert isinstance(close_data, DataCollection)
    assert close_data.is_closed()
    assert output_queue.empty()

    if frame_pool:
        assert frame_pool.is_empty()


@pytest.mark.parametrize('use_frame_pool', [False, True])
def test_produce_tracking_with_video(
        sample_capture_settings: CaptureSettings,
        use_frame_pool: bool
) -> None:
    frame_pool: Optional[FramePool] = create_frame_pool(
        10, sample_capture_settings) if use_frame_pool else None
    frame_queue: 'Queue[DataCollection]' = Queue()
    tracking_queue: 'Queue[DataCollection]' = Queue()

    frame_producer = VideoCaptureProducer(
        frame_queue, sample_capture_settings, frame_pool)
    frame_producer.start()

    produce_tracking(frame_queue, tracking_queue, 1, frame_pool)

    assert tracking_queue.qsize() == 2
    check_tracking_data(tracking_queue.get(), None, frame_pool)

    data: DataCollection = tracking_queue.get()
    assert data.is_closed()
    assert not data.has(ExceptionCloseData), data.get(
        ExceptionCloseData).exception
    assert not data.has(TrackingData)

    frame_producer.stop()
    clear_queue(frame_queue)
    clear_queue(tracking_queue)


@pytest.mark.parametrize('use_frame_pool', [False, True])
def test_producer(
    sample_capture_settings: CaptureSettings,
    use_frame_pool: bool
) -> None:
    frame_pool: Optional[FramePool] = create_frame_pool(
        2, sample_capture_settings) if use_frame_pool else None
    frame_queue: 'Queue[DataCollection]' = Queue()
    tracking_queue: 'Queue[DataCollection]' = Queue()

    frame_producer = VideoCaptureProducer(
        frame_queue, sample_capture_settings, frame_pool)
    frame_producer.start()

    tracking_producer = TrackProducer(
        frame_queue, tracking_queue, 1, frame_pool)
    tracking_producer.start()

    data: Optional[DataCollection] = None
    found_frame = False
    while not found_frame:
        try:
            data = tracking_queue.get(timeout=0.01)
            assert not data.is_closed()
            found_frame = True
        except queue.Empty:
            pass

    assert isinstance(data, DataCollection)
    check_tracking_data(data, sample_capture_settings, frame_pool)
    if frame_pool is not None:
        frame_pool.free_frame(data.get(FrameData).frame)

    frame_producer.stop()
    tracking_producer.stop()
    clear_queue(frame_queue)
    clear_queue(tracking_queue)


def test_produce_tracking_logs(caplog: pytest.LogCaptureFixture) -> None:
    input_queue: 'Queue[DataCollection]' = Queue()
    output_queue: 'Queue[DataCollection]' = Queue()

    input_queue.put(DataCollection().add(
        FrameData(np.zeros((100, 100, 3), dtype=np.uint8))))
    input_queue.put(DataCollection().add(
        FrameData(np.zeros((100, 100, 3), dtype=np.uint8))))
    input_queue.put(DataCollection().add(
        FrameData(np.zeros((100, 100, 3), dtype=np.uint8))))
    input_queue.put(DataCollection().add(
        FrameData(np.zeros((100, 100, 3), dtype=np.uint8))))
    input_queue.put(DataCollection().add(CloseData()))

    with caplog.at_level(logging.INFO):
        produce_tracking(input_queue, output_queue, log_cylces=2)

        assert output_queue.qsize() == 2
        check_tracking_data(output_queue.get(), None)

        close_data = output_queue.get()
        assert isinstance(close_data, DataCollection)
        assert close_data.is_closed()
        assert output_queue.empty()

        log_tuples = caplog.record_tuples
        assert len(log_tuples) == 1
        for log_tuple in log_tuples:
            assert log_tuple[0] == 'root'
            assert log_tuple[1] == logging.INFO
            assert log_tuple[2].startswith('Tracking-FPS:')


def test_stop_track_producer_early() -> None:
    frame_queue: 'Queue[DataCollection]' = Queue()
    tracking_queue: 'Queue[DataCollection]' = Queue()
    track_producer = TrackProducer(frame_queue, tracking_queue)
    track_producer.stop()
