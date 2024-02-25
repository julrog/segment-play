from dataclasses import dataclass
from typing import List

import numpy as np

from pipeline.data import DataCollection
from pose.producer import PoseData
from tracking.producer import TrackingData


@dataclass
class Joint:
    x: float
    y: float
    z: float
    visibility: float


class Landmarks:
    landmark: List[Joint]


def get_verticality(vector: np.ndarray) -> float:
    normalize_vector = vector / np.sqrt(np.sum(vector**2))
    return normalize_vector[1]


def get_x_y_vector(joint_a: Joint, joint_b: Joint) -> np.ndarray:
    return np.array([
        joint_b.x - joint_a.x,
        joint_b.y - joint_a.y
    ], dtype=float)


def detect_narrow_arms(data: DataCollection, id: int) -> bool:
    assert data.has(TrackingData), 'TrackingData is required for this function'
    if data.has(PoseData) and data.get(PoseData).raw_landmarks[id] is not None:
        pose_landmarks = data.get(
            PoseData).raw_landmarks[id].landmark
        upper_right_arm_vec = get_x_y_vector(
            pose_landmarks[12], pose_landmarks[14])
        lower_right_arm_vec = get_x_y_vector(
            pose_landmarks[14], pose_landmarks[16])
        upper_left_arm_vec = get_x_y_vector(
            pose_landmarks[11], pose_landmarks[13])
        lower_left_arm_vec = get_x_y_vector(
            pose_landmarks[13], pose_landmarks[15])
        if get_verticality(upper_right_arm_vec) \
                + get_verticality(lower_right_arm_vec) \
                + get_verticality(upper_left_arm_vec) \
                + get_verticality(lower_left_arm_vec) > 0.8 * 4:
            return True
    else:
        input_box = data.get(TrackingData).get_box(id).astype(np.int32)
        ratio = (input_box[3] - input_box[1]) / \
                (input_box[2] - input_box[0])
        if ratio > 2.5:
            return True
    return False
