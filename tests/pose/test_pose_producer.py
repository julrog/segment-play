import logging
import queue
import time
from multiprocessing import Queue
from typing import Optional

import numpy as np
import pytest

from frame.camera import CaptureSettings
from frame.producer import FrameData, VideoCaptureProducer
from frame.shared import FramePool, create_frame_pool
from pipeline.data import (CloseData, DataCollection, ExceptionCloseData,
                           clear_queue)
from pose.pose import BODY_POINTS, DEFAULT_IMPORTANT_LANDMARKS, Pose
from pose.producer import PoseData, PoseProducer, produce_pose
from segmentation.base import BodyPartSegmentation
from tests.tracking.test_tracking_producer import check_tracking_data
from tracking.producer import TrackingData, TrackProducer, produce_tracking


@pytest.fixture
def pose_data(sample_image: np.ndarray) -> PoseData:
    pose = Pose()
    important_landmarks, raw_landmarks = pose.predict(sample_image)
    return PoseData([important_landmarks], [raw_landmarks])


def check_pose_for_id(
        pose_data: PoseData,
        id: int,
        people_in_picture: bool = True
) -> None:
    landmarks, point_modes = pose_data.get_landmarks_xy(
        id, visibility_threshold=-1.0)
    if people_in_picture:
        assert isinstance(landmarks, np.ndarray)
        assert isinstance(point_modes, list)
        assert len(landmarks) == len(DEFAULT_IMPORTANT_LANDMARKS)
        assert len(point_modes) == len(DEFAULT_IMPORTANT_LANDMARKS)
        assert point_modes == [1.0] * len(DEFAULT_IMPORTANT_LANDMARKS)
    else:
        assert landmarks is None
        assert point_modes is None


def check_pose_data(
        data: DataCollection,
        capture_settings: Optional[CaptureSettings] = None,
        frame_pool: Optional[FramePool] = None,
        people_in_picture: bool = True
) -> None:
    check_tracking_data(data, capture_settings, frame_pool)
    assert data.has(PoseData)
    pose_data: PoseData = data.get(PoseData)
    assert isinstance(pose_data, PoseData)
    check_pose_for_id(pose_data, 0, people_in_picture)


def check_all_pose_data(
        data: DataCollection,
        capture_settings: Optional[CaptureSettings] = None,
        frame_pool: Optional[FramePool] = None
) -> None:
    check_tracking_data(data, capture_settings, frame_pool)
    assert data.has(PoseData)
    pose_data: PoseData = data.get(PoseData)
    assert isinstance(pose_data, PoseData)
    for pose_id in range(len(pose_data.landmarks)):
        check_pose_for_id(pose_data, pose_id, len(
            pose_data.landmarks[pose_id]) > 0)


def test_get_landmarks_xy(pose_data: PoseData) -> None:
    check_pose_for_id(pose_data, 0)


def test_get_landmarks_xy_with_bodypart(pose_data: PoseData) -> None:
    landmarks, point_modes = pose_data.get_landmarks_xy(
        0, specific_bodypart=BodyPartSegmentation.LEFT_ARM)

    left_arm_points = BODY_POINTS[BodyPartSegmentation.LEFT_ARM.value - 1]
    num_points = np.count_nonzero(left_arm_points)
    left_arm_points = list(filter(lambda x: x != 0.0, left_arm_points))

    assert isinstance(landmarks, np.ndarray)
    assert isinstance(point_modes, list)
    assert len(landmarks) == num_points
    assert len(point_modes) == num_points
    assert point_modes == left_arm_points


def test_get_landmarks_xy_with_visibility_threshold(
        pose_data: PoseData) -> None:
    full_landmarks = pose_data.landmarks[0].flatten()[3::4]
    median = np.median(full_landmarks)
    num_above_median = np.count_nonzero(full_landmarks > median)

    landmarks, point_modes = pose_data.get_landmarks_xy(
        0, visibility_threshold=median)
    assert isinstance(landmarks, np.ndarray)
    assert isinstance(point_modes, list)
    assert len(landmarks) == num_above_median
    assert len(point_modes) == num_above_median


def test_get_landmarks_xy_with_no_landmarks(pose_data: PoseData) -> None:
    landmarks, point_modes = pose_data.get_landmarks_xy(1)
    assert landmarks is None
    assert point_modes is None


@pytest.mark.parametrize('use_frame_pool', [False, True])
def test_produce_pose(
        sample_image: np.ndarray,
        use_frame_pool: bool
) -> None:
    frame_pool: Optional[FramePool] = FramePool(
        sample_image, 10) if use_frame_pool else None
    frame_queue: 'Queue[DataCollection]' = Queue()
    tracking_queue: 'Queue[DataCollection]' = Queue()
    pose_queue: 'Queue[DataCollection]' = Queue()

    for _ in range(3):
        frame_queue.put(DataCollection().add(
            FrameData(sample_image, frame_pool)))
    frame_queue.put(DataCollection().add(
        FrameData(np.zeros((1280, 1920, 3), dtype=np.uint8), frame_pool)))
    frame_queue.put(DataCollection().add(CloseData()))

    produce_tracking(frame_queue, tracking_queue, 1,
                     frame_pool, skip_frames=False)

    produce_pose(tracking_queue, pose_queue, 1, frame_pool, skip_frames=False)

    time.sleep(0.1)
    assert frame_queue.empty()
    assert tracking_queue.empty()
    assert pose_queue.qsize() == 5

    for i in range(3 + 1):
        data = pose_queue.get()
        assert isinstance(data, DataCollection)
        check_pose_data(data, None, frame_pool, i < 3)

    data = pose_queue.get()
    assert data.is_closed()
    assert not data.has(ExceptionCloseData), data.get(
        ExceptionCloseData).exception
    assert not data.has(PoseData)

    clear_queue(frame_queue)
    clear_queue(tracking_queue)
    clear_queue(pose_queue)


# TODO: check why it fails with not using frame pool
@pytest.mark.parametrize('use_frame_pool', [True])
def test_produce_pose_with_video(
        short_sample_capture_settings: CaptureSettings,
        use_frame_pool: bool
) -> None:
    frame_pool: Optional[FramePool] = create_frame_pool(
        10, short_sample_capture_settings) if use_frame_pool else None
    frame_queue: 'Queue[DataCollection]' = Queue()
    tracking_queue: 'Queue[DataCollection]' = Queue()
    pose_queue: 'Queue[DataCollection]' = Queue()

    tracking_producer = TrackProducer(
        frame_queue, tracking_queue, 1, frame_pool, skip_frames=False)
    tracking_producer.start()

    frame_producer = VideoCaptureProducer(
        frame_queue,
        short_sample_capture_settings,
        frame_pool,
        skip_frames=False
    )
    frame_producer.start()

    produce_pose(tracking_queue, pose_queue, 1, frame_pool)

    assert pose_queue.qsize() == 2
    check_pose_data(pose_queue.get(), None, frame_pool, False)

    data: DataCollection = pose_queue.get()

    assert data.is_closed()
    assert not data.has(ExceptionCloseData), data.get(
        ExceptionCloseData).exception
    assert not data.has(PoseData)

    frame_producer.stop()
    tracking_producer.stop()
    clear_queue(frame_queue)
    clear_queue(tracking_queue)
    clear_queue(pose_queue)


@pytest.mark.parametrize('use_frame_pool', [False, True])
def test_producer(
    sample_capture_settings: CaptureSettings,
    use_frame_pool: bool
) -> None:
    frame_pool: Optional[FramePool] = create_frame_pool(
        10, sample_capture_settings) if use_frame_pool else None
    frame_queue: 'Queue[DataCollection]' = Queue()
    tracking_queue: 'Queue[DataCollection]' = Queue()
    pose_queue: 'Queue[DataCollection]' = Queue()

    pose_producer = PoseProducer(
        tracking_queue, pose_queue, 1, frame_pool, skip_frames=False)
    pose_producer.start()

    tracking_producer = TrackProducer(
        frame_queue, tracking_queue, 1, frame_pool, skip_frames=False)
    tracking_producer.start()

    frame_producer = VideoCaptureProducer(
        frame_queue, sample_capture_settings, frame_pool, skip_frames=False)
    frame_producer.start()

    data: Optional[DataCollection] = None
    found_frame = False
    while not found_frame:
        try:
            data = pose_queue.get(timeout=0.01)
            if data.is_closed():
                print(data.has(CloseData))
            assert not data.is_closed()
            found_frame = True
        except queue.Empty:
            pass

    assert isinstance(data, DataCollection)
    check_all_pose_data(data, sample_capture_settings, frame_pool)
    if frame_pool is not None:
        frame_pool.free_frame(data.get(FrameData).frame)

    frame_producer.stop()
    tracking_producer.stop()
    pose_producer.stop()
    clear_queue(frame_queue)
    clear_queue(tracking_queue)
    clear_queue(pose_queue)


def test_produce_pose_logs(caplog: pytest.LogCaptureFixture) -> None:
    input_queue: 'Queue[DataCollection]' = Queue()
    output_queue: 'Queue[DataCollection]' = Queue()

    for _ in range(4):
        input_queue.put(DataCollection().add(
            FrameData(np.zeros((100, 100, 3), dtype=np.uint8))
        ).add(
            TrackingData(
                np.array([
                    [0, 0, 100, 100, 1, 20, 20, 80, 80],
                    [0, 0, 100, 100, 2, 20, 20, 80, 80]
                ])))
        )
    input_queue.put(DataCollection().add(CloseData()))

    with caplog.at_level(logging.INFO):
        produce_pose(input_queue, output_queue, log_cylces=2)

        assert output_queue.qsize() == 2
        check_pose_data(output_queue.get(), None, None, False)

        close_data = output_queue.get()
        assert isinstance(close_data, DataCollection)
        assert close_data.is_closed()
        assert output_queue.empty()

        log_tuples = caplog.record_tuples
        assert len(log_tuples) == 1
        for log_tuple in log_tuples:
            assert log_tuple[0] == 'root'
            assert log_tuple[1] == logging.INFO
            assert log_tuple[2].startswith('Pose-FPS:')


def test_stop_pose_producer_early() -> None:
    frame_queue: 'Queue[DataCollection]' = Queue()
    pose_queue: 'Queue[DataCollection]' = Queue()
    track_producer = PoseProducer(frame_queue, pose_queue)
    track_producer.stop()
