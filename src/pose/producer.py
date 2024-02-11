from __future__ import annotations

import logging
import time
from multiprocessing import Process, Queue
from typing import Any, List, Optional, Tuple

import numpy as np

from frame.producer import FrameData, free_output_queue
from frame.shared import FramePool
from ocsort.timer import Timer
from pipeline.data import (BaseData, CloseData, DataCollection,
                           pipeline_data_generator)
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
    input_queue: Queue[DataCollection],
    output_queue: Queue[DataCollection],
    model_complexity: int = 1,
    frame_pool: Optional[FramePool] = None,
    skip_frames: bool = True,
    log_cylces: int = 100
) -> None:
    reduce_frame_discard_timer = 0.0
    timer = Timer()
    pose = Pose(model_complexity)
    frame_count = 0
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
                output_queue, frame_pool, reduce_frame_discard_timer)
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
    if skip_frames:
        free_output_queue(output_queue, frame_pool)
    pose.close()
    input_queue.cancel_join_thread()
    output_queue.cancel_join_thread()


class PoseProducer:
    def __init__(
        self,
        input_queue: Queue[DataCollection],
        output_queue: Queue[DataCollection],
        model_complexity: int = 1,
        frame_pool: Optional[FramePool] = None,
        skip_frames: bool = True,
        log_cycles: int = 100
    ) -> None:
        self.process: Optional[Process] = None
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_complexity = model_complexity
        self.frame_pool = frame_pool
        self.skip_frames = skip_frames
        self.log_cycles = log_cycles

    def start(self) -> None:
        self.process = Process(target=produce_pose, args=(
            self.input_queue,
            self.output_queue,
            self.model_complexity,
            self.frame_pool,
            self.skip_frames,
            self.log_cycles
        ))
        self.process.start()

    def stop(self) -> None:
        self.input_queue.put(DataCollection().add(CloseData()))
        if self.process:
            time.sleep(1)
            self.process.kill()
