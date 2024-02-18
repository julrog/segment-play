from multiprocessing import Queue
from typing import List

import numpy as np
import pytest

from frame.camera import CaptureSettings
from frame.producer import FrameData
from frame.shared import FramePool, create_frame_pool
from pipeline.data import DataCollection
from pipeline.manager import FrameProcessingPipeline, clear_queue
from pose.producer import PoseData
from segmentation.producer import SegmentationData
from tracking.producer import TrackingData


@pytest.mark.parametrize('use_pose', [True, False])
def test_frame_processing_pipeline(
    short_sample_capture_settings: CaptureSettings,
    use_pose: bool
) -> None:
    frame_pool = create_frame_pool(
        100, short_sample_capture_settings)
    pipeline = FrameProcessingPipeline(
        1,
        frame_pool=frame_pool,
        use_pose=use_pose,
        camera_settings=short_sample_capture_settings,
        skip_capture_frames=False
    )
    pipeline.start()

    frames: List[DataCollection] = []
    for frame in pipeline.get_frames():
        if frame.is_closed():
            last_frame = frame
            break
        frames.append(frame)

    assert len(frames) > 1

    for frame in frames:
        assert frame.has(FrameData)
        assert frame.get(TrackingData)
        if use_pose:
            assert frame.get(PoseData)
        assert frame.get(SegmentationData)
        frame_pool.free_frame(frame.get(FrameData).frame)

    assert last_frame.is_closed()

    pipeline.stop()


def test_clear_queue_without_frame_pool() -> None:
    queue: 'Queue[DataCollection]' = Queue()

    queue.put(DataCollection())

    clear_queue(queue)


@pytest.mark.parametrize('frame_data', [True, False])
def test_clear_queue_with_frame_pool(
    frame_data: bool
) -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    frame_pool = FramePool(image, 2)
    queue: 'Queue[DataCollection]' = Queue()

    data = DataCollection()
    if frame_data:
        data.add(FrameData(image, frame_pool))
        assert not frame_pool.is_empty()

    queue.put(data)

    clear_queue(queue, frame_pool)

    assert frame_pool.is_empty()
