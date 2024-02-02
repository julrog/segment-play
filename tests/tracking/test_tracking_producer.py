from typing import Optional

import numpy as np

from frame.camera import CaptureSettings
from frame.shared import FramePool
from pipeline.data import DataCollection
from tests.frame.test_frame_producer import check_frame_data
from tracking.producer import TrackingData


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

        box = tracking_data.get_box()
        assert isinstance(box, np.ndarray)
        assert box.shape == (4,)

        padded_box = tracking_data.get_padded_box()
        assert isinstance(padded_box, np.ndarray)
        assert padded_box.shape == (4,)

        tid = tracking_data.get_tracking_id(id)
        assert isinstance(tid, np.float_)


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


'''@pytest.mark.parametrize('use_frame_pool', [False, True])
def test_produce_tracking(
        sample_capture_settings: CaptureSettings,
        use_frame_pool: bool
) -> None:
    frame_pool: Optional[FramePool] = create_frame_pool(
        2, sample_capture_settings) if use_frame_pool else None
    frame_queue: Queue[DataCollection] = Queue()
    tracking_queue: Queue[DataCollection] = Queue()

    producer = VideoCaptureProducer(
        frame_queue, sample_capture_settings, frame_pool)
    producer.start()

    produce_tracking(frame_queue, tracking_queue, 1, frame_pool)
    data: DataCollection = tracking_queue.get()

    assert data.is_closed()
    assert not data.has(TrackingData)
    # check_tracking_data(data, sample_capture_settings, frame_pool)

    producer.stop()
    clear_queue(frame_queue)
    clear_queue(tracking_queue)'''
