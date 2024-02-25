from typing import Callable, List

import numpy as np
import pytest

from pipeline.data import DataCollection
from pose.analyze import (Joint, Landmarks, detect_narrow_arms,
                          get_verticality, get_x_y_vector)
from pose.producer import PoseData
from tracking.producer import TrackingData


@pytest.mark.parametrize('vector, expected', [
    (np.array([1, 2, 3]), 2 / np.sqrt(14)),
    (np.array([1, 0, 0]), 0),
    (np.array([0, 1, 0]), 1),
    (np.array([0, 0, 1]), 0)
])
def test_get_verticality(vector: np.ndarray, expected: float) -> None:
    assert get_verticality(vector) == expected


def test_get_x_y_vector() -> None:
    joint_a = Joint(1, 2, 0, 0)
    joint_b = Joint(4, 6, 0, 0)
    result = get_x_y_vector(joint_a, joint_b)
    expected = np.array([3, 4], dtype=float)
    assert np.array_equal(result, expected)


def test_detect_narrow_arms(
    pose_data_collection: DataCollection,
) -> None:
    pose_data = pose_data_collection.get(PoseData)
    tracking_data = pose_data_collection.get(TrackingData)

    def set_landmark_data(
        raw_landmarks: List[Landmarks],
        joint_id: int,
        landmarks: List[float],
        id: int = 0
    ) -> None:
        raw_landmarks[id].landmark[joint_id].x = landmarks[0]
        raw_landmarks[id].landmark[joint_id].y = landmarks[1]
        raw_landmarks[id].landmark[joint_id].z = landmarks[2]

    set_landmark_data(pose_data.raw_landmarks, 12, [0.0, 0.0, 0.0])
    set_landmark_data(pose_data.raw_landmarks, 14, [1.0, 1.0, 0.0])
    set_landmark_data(pose_data.raw_landmarks, 16, [2.0, 2.0, 0.0])
    set_landmark_data(pose_data.raw_landmarks, 11, [0.0, 0.0, 0.0])
    set_landmark_data(pose_data.raw_landmarks, 13, [1.0, 1.0, 0.0])
    set_landmark_data(pose_data.raw_landmarks, 15, [2.0, 2.0, 0.0])
    assert not detect_narrow_arms(pose_data_collection, 0)

    set_landmark_data(pose_data.raw_landmarks, 12, [0.0, 0.0, 0.0])
    set_landmark_data(pose_data.raw_landmarks, 14, [0.0, 1.0, 0.0])
    set_landmark_data(pose_data.raw_landmarks, 16, [0.0, 2.0, 0.0])
    set_landmark_data(pose_data.raw_landmarks, 11, [0.0, 0.0, 0.0])
    set_landmark_data(pose_data.raw_landmarks, 13, [0.0, 1.0, 0.0])
    set_landmark_data(pose_data.raw_landmarks, 15, [0.0, 2.0, 0.0])
    assert detect_narrow_arms(pose_data_collection, 0)

    pose_data.raw_landmarks = [None]

    tracking_data.targets = np.array([[0, 0, 100, 100]])
    assert not detect_narrow_arms(pose_data_collection, 0)

    tracking_data.targets = np.array([[0, 0, 10, 100]])
    assert detect_narrow_arms(pose_data_collection, 0)

    with pytest.raises(AssertionError) as e:
        detect_narrow_arms(DataCollection(), 0)
    assert str(e.value) == 'TrackingData is required for this function'


def test_non_narrow_pose(
    pose_data_generator: Callable[[str], DataCollection],
    sample_image_path: str
) -> None:
    pose_data = pose_data_generator(sample_image_path)

    assert not detect_narrow_arms(pose_data, 0)

    pose_data.get(PoseData).raw_landmarks = [None]
    assert not detect_narrow_arms(pose_data, 0)


def test_narrow_pose(
    pose_data_generator: Callable[[str], DataCollection],
    sample_image_2_path: str
) -> None:
    pose_data = pose_data_generator(sample_image_2_path)

    assert detect_narrow_arms(pose_data, 0)

    pose_data.get(PoseData).raw_landmarks = [None]
    assert detect_narrow_arms(pose_data, 0)
