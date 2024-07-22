import logging
import queue
import time
from multiprocessing import Queue, Value
from multiprocessing.sharedctypes import Synchronized
from typing import Optional

import numpy as np
import pytest

from frame.camera import CaptureSettings
from frame.producer import FrameData, VideoCaptureProducer
from frame.shared import FramePool, create_frame_pool
from pipeline.data import CloseData, DataCollection, ExceptionCloseData
from pipeline.manager import clear_queue
from pose.producer import PoseData
from segmentation.base import BodyPartSegmentation
from segmentation.mobile_sam import MobileSam
from segmentation.producer import (SegmentationData, SegmentProducer,
                                   produce_segmentation,
                                   segmentation_calculation)
from tests.tracking.test_tracking_producer import check_tracking_data
from tracking.producer import TrackingData, TrackProducer, produce_tracking
from util.image import create_black_image


def check_segmentation_data_for_id(
        segmentation_data: SegmentationData,
        id: int,
        people_in_picture: bool = True
) -> None:
    mask = segmentation_data.get_mask(id)
    if people_in_picture:
        assert isinstance(mask, np.ndarray)
    else:
        assert mask is None


def check_segmentation_data(
        data: DataCollection,
        capture_settings: Optional[CaptureSettings] = None,
        frame_pool: Optional[FramePool] = None,
        people_in_picture: bool = True
) -> None:
    check_tracking_data(data, capture_settings, frame_pool)
    assert data.has(SegmentationData)
    segmentation_data: SegmentationData = data.get(SegmentationData)
    assert isinstance(segmentation_data, SegmentationData)
    check_segmentation_data_for_id(segmentation_data, 0, people_in_picture)


def check_all_segmentation_data(
        data: DataCollection,
        capture_settings: Optional[CaptureSettings] = None,
        frame_pool: Optional[FramePool] = None
) -> None:
    check_tracking_data(data, capture_settings, frame_pool)
    assert data.has(SegmentationData)
    segmentation_data: SegmentationData = data.get(SegmentationData)
    assert isinstance(segmentation_data, SegmentationData)
    for mask_id in range(len(segmentation_data.masks)):
        check_segmentation_data_for_id(segmentation_data, mask_id, len(
            segmentation_data.masks[mask_id]) > 0)


def test_segmentation(
        segmentation_data_collection: DataCollection) -> None:
    check_segmentation_data_for_id(
        segmentation_data_collection.get(SegmentationData), 0)


def test_segmentation_no_data(
        segmentation_data_collection: DataCollection) -> None:
    segmentation_data = segmentation_data_collection.get(SegmentationData)
    assert segmentation_data.get_mask(1) is None


@pytest.mark.parametrize('pose_data_type', ['None', 'Regular', 'Empty'])
@pytest.mark.parametrize('down_scale', [1.0, 2.0])
def test_segmentation_calculation(
    pose_data_type: str,
    down_scale: float,
    pose_data_collection: DataCollection,
    segmentation_data_collection: DataCollection
) -> None:
    pose_data = None
    if pose_data_type == 'Regular':
        pose_data = pose_data_collection.get(PoseData)
    if pose_data_type == 'Empty':
        pose_data = PoseData([], [])
    segment = MobileSam()
    segmentation_data = segmentation_calculation(
        segment,
        pose_data_collection.get(FrameData).get_frame(),
        down_scale,
        pose_data_collection.get(TrackingData),
        pose_data,
        None
    )
    check_segmentation_data_for_id(segmentation_data, 0)
    assert isinstance(segmentation_data, SegmentationData)
    assert segmentation_data.mask_scale == down_scale
    if down_scale != 1.0:
        regular_mask = segmentation_data_collection.get(
            SegmentationData).masks[0][0]
        down_scale_mask = segmentation_data.masks[0][0]
        assert regular_mask.shape[0] > down_scale_mask.shape[0] * \
            (down_scale * 0.9)
        assert regular_mask.shape[1] > down_scale_mask.shape[1] * \
            (down_scale * 0.9)


@pytest.mark.parametrize('body_parts', [
    None,
    BodyPartSegmentation.ALL,
    BodyPartSegmentation.LEFT_ARM
])
@pytest.mark.parametrize('missing_landmarks', [False, True])
def test_segmentation_calculation_bodyparts(
    body_parts: Optional[BodyPartSegmentation],
    missing_landmarks: bool,
    pose_data_collection: DataCollection,
    segmentation_data_collection: DataCollection
) -> None:
    pose_data = pose_data_collection.get(PoseData)
    segment = MobileSam()
    segmentation_data = segmentation_calculation(
        segment,
        pose_data_collection.get(FrameData).get_frame(),
        1.0,
        pose_data_collection.get(TrackingData),
        pose_data,
        body_parts,
        1.0 if missing_landmarks else 0.5
    )
    check_segmentation_data_for_id(segmentation_data, 0)
    assert isinstance(segmentation_data, SegmentationData)
    assert segmentation_data.mask_scale == 1.0
    if body_parts == BodyPartSegmentation.LEFT_ARM and not missing_landmarks:
        regular_mask = segmentation_data_collection.get(
            SegmentationData).masks[0][0]
        body_part_mask = segmentation_data.masks[0][0]
        assert np.count_nonzero(
            regular_mask) * 0.2 > np.count_nonzero(body_part_mask)


@pytest.mark.parametrize('fast_segmentation', [False, True])
@pytest.mark.parametrize('use_frame_pool', [False, True])
@pytest.mark.parametrize('down_scale', [1.0, 2.0])
def test_produce_segmentation(
        sample_image: np.ndarray,
        use_frame_pool: bool,
        fast_segmentation: bool,
        down_scale: float
) -> None:
    frame_pool: Optional[FramePool] = FramePool(
        sample_image, 10) if use_frame_pool else None
    frame_pools = {FrameData: frame_pool}
    frame_queue: 'Queue[DataCollection]' = Queue()
    tracking_queue: 'Queue[DataCollection]' = Queue()
    segmentation_queue: 'Queue[DataCollection]' = Queue()
    ready_tracking: Synchronized[int] = Value('i', 0)  # type: ignore
    ready: Synchronized[int] = Value('i', 0)  # type: ignore

    for _ in range(3):
        frame_queue.put(DataCollection().add(
            FrameData(sample_image, frame_pool)))
    frame_queue.put(DataCollection().add(
        FrameData(np.zeros((1280, 1920, 3), dtype=np.uint8), frame_pool)))
    frame_queue.put(DataCollection().add(CloseData()))

    produce_tracking(frame_queue, tracking_queue, ready_tracking,
                     frame_pool, skip_frames=False)

    produce_segmentation(
        tracking_queue,
        segmentation_queue,
        ready,
        frame_pool,
        skip_frames=False,
        down_scale=down_scale,
        fast=fast_segmentation,
    )

    time.sleep(0.1)
    assert frame_queue.empty()
    assert tracking_queue.empty()
    assert segmentation_queue.qsize() == 5

    for _ in range(3 + 1):
        data = segmentation_queue.get()
        assert isinstance(data, DataCollection)
        check_segmentation_data(data, None, frame_pool)

    data = segmentation_queue.get()
    assert data.is_closed()
    assert not data.has(ExceptionCloseData), data.get(
        ExceptionCloseData).exception
    assert not data.has(SegmentationData)

    clear_queue(frame_queue, frame_pools)
    clear_queue(tracking_queue, frame_pools)
    clear_queue(segmentation_queue, frame_pools)


# TODO: check why it fails with not using frame pool
@pytest.mark.parametrize('use_frame_pool', [True])
def test_produce_segmentation_with_video(
    short_sample_capture_settings: CaptureSettings,
    use_frame_pool: bool
) -> None:
    frame_pool: Optional[FramePool] = create_frame_pool(
        10, short_sample_capture_settings) if use_frame_pool else None
    frame_pools = {FrameData: frame_pool}
    frame_queue: 'Queue[DataCollection]' = Queue()
    tracking_queue: 'Queue[DataCollection]' = Queue()
    segmentation_queue: 'Queue[DataCollection]' = Queue()
    ready: Synchronized[int] = Value('i', 0)  # type: ignore

    tracking_producer = TrackProducer(
        frame_queue,
        tracking_queue,
        frame_pool,
        skip_frames=False
    )
    tracking_producer.start()

    frame_producer = VideoCaptureProducer(
        frame_queue,
        short_sample_capture_settings,
        frame_pool,
        skip_frames=False
    )
    frame_producer.start()

    produce_segmentation(
        tracking_queue,
        segmentation_queue,
        ready,
        frame_pool,
        down_scale=1,
        fast=True,
    )

    assert segmentation_queue.qsize() == 2
    check_segmentation_data(segmentation_queue.get(), None, frame_pool)

    data: DataCollection = segmentation_queue.get()
    assert data.is_closed()
    assert not data.has(ExceptionCloseData), data.get(
        ExceptionCloseData).exception
    assert not data.has(SegmentationData)

    frame_producer.stop()
    tracking_producer.join()
    clear_queue(frame_queue, frame_pools)
    clear_queue(tracking_queue, frame_pools)
    clear_queue(segmentation_queue, frame_pools)


@pytest.mark.parametrize('use_frame_pool', [True])
def test_producer(
    sample_capture_settings: CaptureSettings,
    use_frame_pool: bool
) -> None:
    frame_pool: Optional[FramePool] = create_frame_pool(
        10, sample_capture_settings) if use_frame_pool else None
    frame_pools = {FrameData: frame_pool}
    frame_queue: 'Queue[DataCollection]' = Queue()
    tracking_queue: 'Queue[DataCollection]' = Queue()
    segmentation_queue: 'Queue[DataCollection]' = Queue()

    segmentation_producer = SegmentProducer(
        tracking_queue,
        segmentation_queue,
        frame_pool,
        skip_frames=False,
        down_scale=1,
        fast=True
    )
    segmentation_producer.start()

    tracking_producer = TrackProducer(
        frame_queue,
        tracking_queue,
        frame_pool,
        skip_frames=False
    )
    tracking_producer.start()

    frame_producer = VideoCaptureProducer(
        frame_queue,
        sample_capture_settings,
        frame_pool,
        skip_frames=False
    )
    frame_producer.start()

    data: Optional[DataCollection] = None
    found_frame = False
    while not found_frame:
        try:
            data = segmentation_queue.get(timeout=0.01)
            if data.is_closed():
                print(data.has(CloseData))
            assert not data.is_closed()
            found_frame = True
        except queue.Empty:
            pass

    assert isinstance(data, DataCollection)
    check_all_segmentation_data(
        data, sample_capture_settings, frame_pool)
    if frame_pool is not None:
        frame_pool.free_frame(data.get(FrameData).frame)

    frame_producer.stop()
    tracking_producer.join()
    segmentation_producer.join()
    clear_queue(frame_queue, frame_pools)
    clear_queue(tracking_queue, frame_pools)
    clear_queue(segmentation_queue, frame_pools)


def test_produce_segmentation_logs(caplog: pytest.LogCaptureFixture) -> None:
    input_queue: 'Queue[DataCollection]' = Queue()
    output_queue: 'Queue[DataCollection]' = Queue()
    ready: Synchronized[int] = Value('i', 0)  # type: ignore

    for _ in range(4):
        input_queue.put(DataCollection().add(
            FrameData(create_black_image((100, 100, 3)))
        ).add(
            TrackingData(
                np.array([
                    [0, 0, 100, 100, 1, 20, 20, 80, 80],
                    [0, 0, 100, 100, 2, 20, 20, 80, 80]
                ])))
        )
    input_queue.put(DataCollection().add(CloseData()))

    with caplog.at_level(logging.INFO):
        produce_segmentation(input_queue, output_queue, ready, log_cylces=2)

        assert output_queue.qsize() == 2
        check_segmentation_data(output_queue.get(), None, None)

        close_data = output_queue.get()
        assert isinstance(close_data, DataCollection)
        assert close_data.is_closed()
        assert output_queue.empty()

        log_tuples = caplog.record_tuples
        assert len(log_tuples) == 1
        for log_tuple in log_tuples:
            assert log_tuple[0] == 'root'
            assert log_tuple[1] == logging.INFO
            assert log_tuple[2].startswith('Segmentation-FPS:')
    clear_queue(input_queue)
    clear_queue(output_queue)


def test_stop_producer() -> None:
    in_queue: 'Queue[DataCollection]' = Queue()
    out_queue: 'Queue[DataCollection]' = Queue()
    producer = SegmentProducer(in_queue, out_queue)
    producer.start()

    while producer.ready.value != 1:
        time.sleep(0.1)

    producer.stop()

    assert in_queue.empty()

    close_data = out_queue.get()
    assert isinstance(close_data, DataCollection)
    assert close_data.is_closed()
    assert out_queue.empty()

    clear_queue(in_queue)
    clear_queue(out_queue)


def test_stop_producer_early() -> None:
    in_queue: 'Queue[DataCollection]' = Queue()
    out_queue: 'Queue[DataCollection]' = Queue()
    producer = SegmentProducer(in_queue, out_queue)
    producer.stop()
    clear_queue(in_queue)
    clear_queue(out_queue)


def test_join_producer_early() -> None:
    in_queue: 'Queue[DataCollection]' = Queue()
    out_queue: 'Queue[DataCollection]' = Queue()
    producer = SegmentProducer(in_queue, out_queue)
    producer.join()
    clear_queue(in_queue)
    clear_queue(out_queue)
