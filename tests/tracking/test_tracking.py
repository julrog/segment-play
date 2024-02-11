from __future__ import annotations

from collections.abc import Callable
from typing import Generator, List, Optional

import numpy as np
import pytest

from tracking.tracking import Tracker, TrackObject
from util.image import create_black_image, scale_image


@pytest.mark.parametrize('appearances', [None, [], [1], [1, 2, 3]])
def test_track_object(appearances: Optional[List[int]]) -> None:
    if appearances is not None:
        track_object = TrackObject(5, appearances, 6)
        assert track_object.last_appearance == appearances
    else:
        track_object = TrackObject(appearance_count=5, last_frame=6)
        assert track_object.last_appearance == [0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_detection(sample_image: np.ndarray) -> None:
    tracker = Tracker()

    detections, img_info = tracker.inference(sample_image)
    assert detections is not None
    assert len(detections[0]) == 5
    assert img_info['height'] == sample_image.shape[0]
    assert img_info['width'] == sample_image.shape[1]
    assert (img_info['raw_img'] == sample_image).all()


def test_detection_basic(
    sample_video_frame_gen: Callable[[], Generator[np.ndarray, None, None]]
) -> None:
    tracker = Tracker()

    for frame in sample_video_frame_gen():
        detections, img_info = tracker.inference(frame)
        assert detections is not None
        assert len(detections[0]) == 5
        assert img_info['height'] == frame.shape[0]
        assert img_info['width'] == frame.shape[1]
        assert (img_info['raw_img'] == frame).all()
        break


def test_detection_no_detections() -> None:
    frame = create_black_image((800, 600, 3))
    tracker = Tracker()

    detections, img_info = tracker.inference(frame)
    assert detections is None
    assert img_info['height'] == frame.shape[0]
    assert img_info['width'] == frame.shape[1]
    assert (img_info['raw_img'] == frame).all()


@pytest.mark.parametrize('down_scale', [1.0, 2.0])
def test_tracker_basic(
    sample_video_frame_gen: Callable[[], Generator[np.ndarray, None, None]],
    down_scale: float
) -> None:
    tracker = Tracker(down_scale)

    count = 0
    for frame in sample_video_frame_gen():

        tracker.update(scale_image(frame, 1.0 / down_scale))
        assert tracker.current_frame == count + 1
        assert tracker.get_all_targets()
        assert tracker.track_objects != {}
        if count < 5:
            assert tracker.track_objects[1].last_frame == count
            assert len(tracker.track_objects[1].last_appearance) == 9
            assert tracker.track_objects[1].appearance_count == count + 1

        count += 1
        if count >= 40:
            break
    assert max(tracker.track_objects) == 4


def test_tracker_no_detections() -> None:
    frame = create_black_image((800, 600, 3))
    tracker = Tracker()

    tracker.update(frame)
    assert tracker.current_frame == 1
    assert tracker.get_all_targets() == []
    assert tracker.track_objects == {}

    tracker.update(frame)
    assert tracker.current_frame == 2
    assert tracker.get_all_targets() == []
    assert tracker.track_objects == {}


def test_narrow_filter(
    sample_video_frame_gen: Callable[[], Generator[np.ndarray, None, None]]
) -> None:
    tracker = Tracker()
    tracker.max_narrowness = 0.1

    count = 0
    for frame in sample_video_frame_gen():
        tracker.update(frame)
        count += 1
        if count >= 20:
            break
    assert len(tracker.track_objects) == 0
