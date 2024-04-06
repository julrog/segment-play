'''from multiprocessing.sharedctypes import Synchronized
from typing import Optional

import numpy as np
import pytest

from background import Background, BackgroundHandle, BackgroundProcessor
from frame.camera import CaptureSettings
from frame.producer import FrameData
from frame.shared import FramePool
from masking.frame_modification import ModifiedFrameData
from masking.producer import produce_masking
from pipeline.data import CloseData, DataCollection, ExceptionCloseData
from pipeline.manager import clear_queue
from segmentation.producer import produce_segmentation
from settings import GameSettings
from tests.segmentation.test_segmentation_producer import \
    check_segmentation_data
from tracking.producer import produce_tracking


def check_modified_frame_data(
        data: DataCollection,
        capture_settings: Optional[CaptureSettings] = None,
        frame_pool: Optional[FramePool] = None
) -> None:
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
        sample_image, 10)
    frame_queue: 'Queue[DataCollection]' = Queue()
    tracking_queue: 'Queue[DataCollection]' = Queue()
    segmentation_queue: 'Queue[DataCollection]' = Queue()
    masking_queue: 'Queue[DataCollection]' = Queue()
    ready_tracking: Synchronized[int] = Value('i', 0)  # type: ignore
    ready: Synchronized[int] = Value('i', 0)  # type: ignore

    bg_processor = BackgroundProcessor(frame_queue, frame_pool, bg_frame_pool)
    bg_processor.start()
    background_handle = BackgroundHandle(bg_processor.background_id,
                                         bg_processor.update_count,
                                         bg_frame_pool, frame_queue)
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

    data = segmentation_queue.get()
    assert data.is_closed()
    assert not data.has(ExceptionCloseData), data.get(
        ExceptionCloseData).exception
    assert not data.has(ModifiedFrameData)

    clear_queue(frame_queue, frame_pools)
    clear_queue(tracking_queue, frame_pools)
    clear_queue(segmentation_queue, frame_pools)
    clear_queue(masking_queue, frame_pools)

    bg_processor.stop()


'''

'''  # TODO: check why it fails with not using frame pool


@pytest.mark.parametrize('use_frame_pool', [True])
def test_produce_segmentation_with_video(
    short_sample_capture_settings: CaptureSettings,
    use_frame_pool: bool
) -> None:
    frame_pool: Optional[FramePool] = create_frame_pool(
        10, short_sample_capture_settings) if use_frame_pool else None
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
    clear_queue(frame_queue, frame_pool)
    clear_queue(tracking_queue, frame_pool)
    clear_queue(segmentation_queue, frame_pool)


@pytest.mark.parametrize('use_frame_pool', [False, True])
def test_producer(
    sample_capture_settings: CaptureSettings,
    use_frame_pool: bool
) -> None:
    frame_pool: Optional[FramePool] = create_frame_pool(
        10, sample_capture_settings) if use_frame_pool else None
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
    clear_queue(frame_queue, frame_pool)
    clear_queue(tracking_queue, frame_pool)
    clear_queue(segmentation_queue, frame_pool)


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


def test_join_producer_early() -> None:
    in_queue: 'Queue[DataCollection]' = Queue()
    out_queue: 'Queue[DataCollection]' = Queue()
    producer = SegmentProducer(in_queue, out_queue)
    producer.join()'''
