import logging
import queue
import time
from multiprocessing import Queue
from multiprocessing.sharedctypes import Synchronized, Value
from typing import Dict, Optional, Tuple

import numpy as np
import pytest

from background import BackgroundHandle, BackgroundProcessor
from frame.camera import CaptureSettings
from frame.producer import FrameData, VideoCaptureProducer
from frame.shared import FramePool, create_frame_pool
from masking.frame_modification import ModifiedFrameData
from masking.producer import MaskingProducer, produce_masking
from pipeline.data import CloseData, DataCollection, ExceptionCloseData
from pipeline.manager import clear_queue
from segmentation.producer import (SegmentationData, SegmentProducer,
                                   produce_segmentation)
from settings import GameSettings
from tests.segmentation.test_segmentation_producer import \
    check_segmentation_data
from tracking.producer import TrackingData, TrackProducer, produce_tracking
from util.image import create_black_image


def check_modified_frame_data(
        data: DataCollection,
        capture_settings: Optional[CaptureSettings] = None,
        frame_pool: Optional[FramePool] = None,
        segmentation_check: bool = True
) -> None:
    if segmentation_check:
        check_segmentation_data(data, capture_settings, frame_pool)
    assert data.has(ModifiedFrameData)
    modified_frame_data: ModifiedFrameData = data.get(ModifiedFrameData)
    assert isinstance(modified_frame_data, ModifiedFrameData)
    assert modified_frame_data.frame is not None
    if frame_pool is not None:
        assert isinstance(modified_frame_data.frame, int)
    else:
        assert isinstance(modified_frame_data.frame, np.ndarray)


@pytest.mark.parametrize('test_settings', [
    ('original', {'all_invisibility': False, 'overall_mirror': False}),
])
@pytest.mark.parametrize('use_frame_pool', [False, True])
def test_produce_masking(
        sample_image: np.ndarray,
        test_settings: Tuple[str, Dict[str, bool]],
        use_frame_pool: bool,
) -> None:
    frame_pool: Optional[FramePool] = FramePool(
        sample_image, 10) if use_frame_pool else None
    modified_frame_pool: Optional[FramePool] = FramePool(
        sample_image, 10) if use_frame_pool else None
    frame_pools = {FrameData: frame_pool,
                   ModifiedFrameData: modified_frame_pool}
    bg_frame_pool: FramePool = FramePool(
        sample_image.astype(np.float32), 10)
    frame_queue: 'Queue[DataCollection]' = Queue()
    tracking_queue: 'Queue[DataCollection]' = Queue()
    segmentation_queue: 'Queue[DataCollection]' = Queue()
    masking_queue: 'Queue[DataCollection]' = Queue()
    ready_masking: Synchronized[int] = Value('i', 0)  # type: ignore
    ready_tracking: Synchronized[int] = Value('i', 0)  # type: ignore
    ready: Synchronized[int] = Value('i', 0)  # type: ignore

    background_queue: 'Queue[DataCollection]' = Queue()
    bg_processor = BackgroundProcessor(
        background_queue, frame_pool, bg_frame_pool)
    bg_processor.start()
    background_handle = BackgroundHandle(bg_processor.background_id,
                                         bg_processor.update_count,
                                         bg_frame_pool, background_queue)
    setting = GameSettings()
    _, overwrite_setting = test_settings
    for key, value in overwrite_setting.items():
        setting.set(key, value)

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
        fast=False,
    )

    produce_masking(
        segmentation_queue,
        masking_queue,
        ready_masking,
        setting,
        background_handle,
        frame_pool,
        modified_frame_pool,
        skip_frames=False,
    )

    time.sleep(0.1)
    assert frame_queue.empty()
    assert tracking_queue.empty()
    assert segmentation_queue.empty()
    assert masking_queue.qsize() == 5

    for _ in range(3 + 1):
        data = masking_queue.get()
        assert isinstance(data, DataCollection)
        check_modified_frame_data(data, None, frame_pool)

    data = masking_queue.get()
    assert data.is_closed()
    assert not data.has(ExceptionCloseData), data.get(
        ExceptionCloseData).exception
    assert not data.has(ModifiedFrameData)

    clear_queue(frame_queue, frame_pools)
    clear_queue(tracking_queue, frame_pools)
    clear_queue(segmentation_queue, frame_pools)
    clear_queue(masking_queue, frame_pools)

    bg_processor.stop()
    clear_queue(background_queue, {FrameData: bg_frame_pool})


# TODO: check why it fails with not using frame pool
@pytest.mark.parametrize('use_frame_pool', [True])
def test_produce_masking_with_video(
    short_sample_capture_settings: CaptureSettings,
    use_frame_pool: bool
) -> None:
    frame_pool: Optional[FramePool] = create_frame_pool(
        10, short_sample_capture_settings) if use_frame_pool else None
    frame_pool_shape = frame_pool.shape if isinstance(
        frame_pool, FramePool) else [100, 100, 3]
    modified_frame_pool: Optional[FramePool] = FramePool(
        create_black_image(frame_pool_shape), 10) if use_frame_pool else None
    frame_pools = {FrameData: frame_pool,
                   ModifiedFrameData: modified_frame_pool}
    bg_frame_pool: FramePool = FramePool(
        create_black_image(frame_pool_shape).astype(np.float32), 10)
    frame_queue: 'Queue[DataCollection]' = Queue()
    tracking_queue: 'Queue[DataCollection]' = Queue()
    segmentation_queue: 'Queue[DataCollection]' = Queue()
    masking_queue: 'Queue[DataCollection]' = Queue()

    background_queue: 'Queue[DataCollection]' = Queue()
    bg_processor = BackgroundProcessor(
        background_queue, frame_pool, bg_frame_pool)
    bg_processor.start()
    background_handle = BackgroundHandle(bg_processor.background_id,
                                         bg_processor.update_count,
                                         bg_frame_pool, background_queue)
    setting = GameSettings()
    ready_masking: Synchronized[int] = Value('i', 0)  # type: ignore

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
        short_sample_capture_settings,
        frame_pool,
        skip_frames=False
    )
    frame_producer.start()

    produce_masking(
        segmentation_queue,
        masking_queue,
        ready_masking,
        setting,
        background_handle,
        frame_pool,
        modified_frame_pool
    )

    assert masking_queue.qsize() == 2
    check_modified_frame_data(masking_queue.get(), None, frame_pool)

    data: DataCollection = masking_queue.get()
    assert data.is_closed()
    assert not data.has(ExceptionCloseData), data.get(
        ExceptionCloseData).exception
    assert not data.has(ModifiedFrameData)

    frame_producer.stop()
    tracking_producer.join()
    segmentation_producer.join()
    clear_queue(frame_queue, frame_pools)
    clear_queue(tracking_queue, frame_pools)
    clear_queue(segmentation_queue, frame_pools)
    clear_queue(masking_queue, frame_pools)

    bg_processor.stop()
    clear_queue(background_queue, {FrameData: bg_frame_pool})


@pytest.mark.parametrize('use_frame_pool', [True])
def test_producer(
    sample_capture_settings: CaptureSettings,
    use_frame_pool: bool
) -> None:
    frame_pool: Optional[FramePool] = create_frame_pool(
        10, sample_capture_settings) if use_frame_pool else None
    frame_pool_shape = frame_pool.shape if isinstance(
        frame_pool, FramePool) else [100, 100, 3]
    modified_frame_pool: Optional[FramePool] = FramePool(
        create_black_image(frame_pool_shape), 10) if use_frame_pool else None
    frame_pools = {FrameData: frame_pool,
                   ModifiedFrameData: modified_frame_pool}
    bg_frame_pool: FramePool = FramePool(
        create_black_image(frame_pool_shape).astype(np.float32), 10)
    frame_queue: 'Queue[DataCollection]' = Queue()
    tracking_queue: 'Queue[DataCollection]' = Queue()
    segmentation_queue: 'Queue[DataCollection]' = Queue()
    masking_queue: 'Queue[DataCollection]' = Queue()

    background_queue: 'Queue[DataCollection]' = Queue()
    bg_processor = BackgroundProcessor(
        background_queue, frame_pool, bg_frame_pool)
    bg_processor.start()
    background_handle = BackgroundHandle(bg_processor.background_id,
                                         bg_processor.update_count,
                                         bg_frame_pool, background_queue)
    setting = GameSettings()

    masking_producer = MaskingProducer(
        segmentation_queue,
        masking_queue,
        setting,
        background_handle,
        frame_pool,
        modified_frame_pool,
        skip_frames=False
    )
    masking_producer.start()

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
            data = masking_queue.get(timeout=0.01)
            if data.is_closed():
                print(data.has(CloseData))
            assert not data.is_closed()
            found_frame = True
        except queue.Empty:
            pass

    assert isinstance(data, DataCollection)
    check_modified_frame_data(
        data, sample_capture_settings, frame_pool)
    if frame_pool is not None:
        frame_pool.free_frame(data.get(FrameData).frame)

    frame_producer.stop()
    tracking_producer.join()
    segmentation_producer.join()
    masking_producer.join()
    clear_queue(frame_queue, frame_pools)
    clear_queue(tracking_queue, frame_pools)
    clear_queue(segmentation_queue, frame_pools)
    clear_queue(masking_queue, frame_pools)

    bg_processor.stop()
    clear_queue(background_queue, {FrameData: bg_frame_pool})


def test_produce_masking_logs(caplog: pytest.LogCaptureFixture) -> None:
    input_queue: 'Queue[DataCollection]' = Queue()
    output_queue: 'Queue[DataCollection]' = Queue()

    setting = GameSettings()
    bg_frame_pool: FramePool = FramePool(
        create_black_image([100, 100, 3]).astype(np.float32), 10)
    background_queue: 'Queue[DataCollection]' = Queue()
    bg_processor = BackgroundProcessor(
        background_queue, None, bg_frame_pool)
    bg_processor.start()
    background_handle = BackgroundHandle(bg_processor.background_id,
                                         bg_processor.update_count,
                                         bg_frame_pool, background_queue)
    ready_masking: Synchronized[int] = Value('i', 0)  # type: ignore

    for _ in range(4):
        input_queue.put(
            DataCollection().add(
                FrameData(create_black_image((100, 100, 3)))
            ).add(
                TrackingData(
                    np.array([
                        [0, 0, 100, 100, 1, 20, 20, 80, 80],
                        [0, 0, 100, 100, 2, 20, 20, 80, 80]
                    ]))
            ).add(
                SegmentationData([[np.array([]), np.array([])]])
            )
        )
    input_queue.put(DataCollection().add(CloseData()))

    with caplog.at_level(logging.INFO):
        produce_masking(input_queue, output_queue, ready_masking, setting,
                        background_handle, None, None, log_cylces=2)

        assert output_queue.qsize() == 2
        check_modified_frame_data(output_queue.get(), None, None, False)

        close_data = output_queue.get()
        assert isinstance(close_data, DataCollection)
        assert close_data.is_closed()
        assert output_queue.empty()

        log_tuples = caplog.record_tuples
        assert len(log_tuples) == 1
        for log_tuple in log_tuples:
            assert log_tuple[0] == 'root'
            assert log_tuple[1] == logging.INFO
            assert log_tuple[2].startswith('Masking-FPS:')

    bg_processor.stop()
    clear_queue(background_queue, {FrameData: bg_frame_pool})


def test_stop_producer() -> None:
    bg_frame_pool: FramePool = FramePool(
        create_black_image([100, 100, 3]).astype(np.float32), 10)

    background_queue: 'Queue[DataCollection]' = Queue()
    bg_processor = BackgroundProcessor(
        background_queue, None, bg_frame_pool)
    bg_processor.start()
    background_handle = BackgroundHandle(bg_processor.background_id,
                                         bg_processor.update_count,
                                         bg_frame_pool, background_queue)
    setting = GameSettings()

    in_queue: 'Queue[DataCollection]' = Queue()
    out_queue: 'Queue[DataCollection]' = Queue()
    producer = MaskingProducer(in_queue, out_queue, setting, background_handle)
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

    bg_processor.stop()
    clear_queue(background_queue, {FrameData: bg_frame_pool})


def test_join_producer_early() -> None:
    bg_frame_pool: FramePool = FramePool(
        create_black_image([100, 100, 3]).astype(np.float32), 10)

    background_queue: 'Queue[DataCollection]' = Queue()
    bg_processor = BackgroundProcessor(
        background_queue, None, bg_frame_pool)
    bg_processor.start()
    background_handle = BackgroundHandle(bg_processor.background_id,
                                         bg_processor.update_count,
                                         bg_frame_pool, background_queue)
    setting = GameSettings()

    in_queue: 'Queue[DataCollection]' = Queue()
    out_queue: 'Queue[DataCollection]' = Queue()
    producer = MaskingProducer(in_queue, out_queue, setting, background_handle)
    producer.start()

    producer.join()

    clear_queue(in_queue)
    clear_queue(out_queue)

    bg_processor.stop()
    clear_queue(background_queue, {FrameData: bg_frame_pool})
