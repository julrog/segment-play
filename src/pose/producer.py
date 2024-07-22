from __future__ import annotations

import logging
import time
from multiprocessing import Queue, Value
from multiprocessing.sharedctypes import Synchronized
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from frame.producer import FrameData, free_output_queue
from frame.shared import FramePool
from ocsort.timer import Timer
from pipeline.data import (BaseData, DataCollection, ExceptionCloseData,
                           pipeline_data_generator)
from pipeline.producer import Producer
from pose.pose import BODY_POINTS, Pose
from segmentation.base import BodyPartSegmentation
from tracking.producer import TrackingData


class PoseData(BaseData):
    def __init__(
        self,
        landmarks: List[np.ndarray],
        raw_landmarks: List[Any],
    ) -> None:
        super().__init__()
        self.landmarks = landmarks
        self.raw_landmarks = raw_landmarks

    def get_landmarks_xy(
        self,
        id: int,
        specific_bodypart: BodyPartSegmentation = BodyPartSegmentation.ALL,
        visibility_threshold: float = 0.5
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if id >= len(self.landmarks) or not self.landmarks[id].any():
            return None, None
        point_modes = []
        landmarks = []
        for landmark_id, landmark in enumerate(self.landmarks[id]):
            if landmark[3] > visibility_threshold:
                if specific_bodypart != BodyPartSegmentation.ALL:
                    points = BODY_POINTS[specific_bodypart.value - 1]
                    if points[landmark_id] != 0.0:
                        point_modes.append(points[landmark_id])
                        landmarks.append(self.landmarks[id][landmark_id, :2])
                else:
                    point_modes.append(1.0)
                    landmarks.append(self.landmarks[id][landmark_id, :2])
        return np.array(landmarks), point_modes


def region_pose_estimation(
        pose: Pose,
        tracking_data: TrackingData,
        frame: np.ndarray
) -> PoseData:
    all_landmarks = []
    all_raw_landmarks = []
    for id in range(len(tracking_data.targets)):
        pad_box = tracking_data.get_padded_box(id)
        cropped_conv_frame = \
            frame[int(pad_box[1]):int(pad_box[3]),
                  int(pad_box[0]):int(pad_box[2])]
        landmarks, raw_landmarks = pose.predict(cropped_conv_frame)

        if landmarks.any():
            landmarks[:, 0] += pad_box[0]
            landmarks[:, 1] += pad_box[1]
        all_landmarks.append(landmarks)
        all_raw_landmarks.append(raw_landmarks)
    return PoseData(all_landmarks, all_raw_landmarks)


def produce_pose(
    input_queue: 'Queue[DataCollection]',
    output_queue: 'Queue[DataCollection]',
    ready: 'Synchronized[int]',
    frame_pool: Optional[FramePool] = None,
    skip_frames: bool = True,
    log_cylces: int = 100,
    model_complexity: int = 1,
) -> None:
    try:
        frame_pools: Dict[Type, Optional[FramePool]] = {FrameData: frame_pool}
        reduce_frame_discard_timer = 0.0
        timer = Timer()
        pose = Pose(model_complexity)
        frame_count = 0
        ready.value = 1

        for data in pipeline_data_generator(
            input_queue,
            output_queue,
            [TrackingData]
        ):
            timer.tic()
            frame = data.get(FrameData).get_frame(frame_pool)

            pose_data = region_pose_estimation(
                pose, data.get(TrackingData), frame)

            if skip_frames:
                reduce_frame_discard_timer = free_output_queue(
                    output_queue, frame_pools, reduce_frame_discard_timer)
            output_queue.put(data.add(pose_data))

            timer.toc()
            frame_count += 1
            if frame_count == log_cylces:
                timer.clear()
            if frame_count % log_cylces == 0 and frame_count > log_cylces:
                average_time = 1. / timer.average_time, 1. / \
                    (timer.average_time + reduce_frame_discard_timer)
                logging.info(f'Pose-FPS: {average_time}')
            if skip_frames and reduce_frame_discard_timer > 0.015:
                time.sleep(reduce_frame_discard_timer)
    except Exception as e:  # pragma: no cover
        if skip_frames:
            free_output_queue(output_queue, frame_pools)
        output_queue.put(DataCollection().add(
            ExceptionCloseData(e)))
    if pose:  # pragma: no cover
        pose.close()


class PoseProducer(Producer):
    def __init__(
        self,
        input_queue: 'Queue[DataCollection]',
        output_queue: 'Queue[DataCollection]',
        frame_pool: Optional[FramePool] = None,
        skip_frames: bool = True,
        log_cycles: int = 100,
        model_complexity: int = 1,
    ) -> None:
        self.ready: Synchronized[int] = Value('i', 0)  # type: ignore
        super().__init__(input_queue, output_queue, self.ready, frame_pool,
                         skip_frames, log_cycles, model_complexity)

    def start(self, handle_logs: bool = False) -> None:
        self.base_start(produce_pose, handle_logs)
