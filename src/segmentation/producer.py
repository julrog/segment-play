from __future__ import annotations

import logging
import time
from multiprocessing import Process, Queue
from multiprocessing.sharedctypes import Synchronized
from typing import List, Optional

import numpy as np

from frame.producer import FrameData, free_output_queue
from frame.shared import FramePool
from ocsort.timer import Timer
from pipeline.data import (BaseData, CloseData, DataCollection,
                           ExceptionCloseData, pipeline_data_generator)
from pose.producer import PoseData
from segmentation.base import BodyPartSegmentation, Segmentation
from segmentation.mobile_sam import MobileSam
from segmentation.sam import Sam
from tracking.producer import TrackingData
from util.image import clip_section_xyxy, scale_image


class SegmentationData(BaseData):
    def __init__(
        self,
        masks: List[List[np.ndarray]],
        mask_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.masks = masks
        self.mask_scale = mask_scale

    def get_mask(self, id: int) -> Optional[np.ndarray]:
        if id >= len(self.masks) or not self.masks[id][0].any():
            return None
        return self.masks[id][0]


def segmentation_calculation(
    segment: Segmentation,
    image: np.ndarray,
    down_scale: Optional[float],
    tracking_data: TrackingData,
    pose_data: Optional[PoseData],
    specific_bodypart: Optional[int],
    visiblity_threshold: float = 0.5
) -> SegmentationData:
    segment.set_image(image)
    segment.prepare_prompts(image)
    all_masks = []
    for id in range(len(tracking_data.targets)):
        input_box = tracking_data.get_box(id)
        pad_box = tracking_data.get_padded_box(id)
        if down_scale != 1.0 and down_scale is not None:
            input_box /= down_scale
            pad_box /= down_scale

        landmarks = None
        point_mode = None

        if pose_data is not None:
            bodypart = BodyPartSegmentation.ALL
            if specific_bodypart is not None:
                bodypart = BodyPartSegmentation(specific_bodypart)
            landmarks, point_mode = pose_data.get_landmarks_xy(
                id, bodypart, visiblity_threshold)
            if landmarks is not None:
                if down_scale != 1.0 and down_scale is not None:
                    landmarks /= down_scale

                if specific_bodypart is not None \
                        and specific_bodypart \
                        != BodyPartSegmentation.ALL.value:
                    padding = min(
                        input_box[2] - input_box[0],
                        input_box[3] - input_box[1]
                    )
                    padding *= 0.25
                    positions_x = [landmark[0] for landmark,
                                   pm in zip(landmarks, point_mode) if pm == 1]
                    positions_y = [landmark[1] for landmark,
                                   pm in zip(landmarks, point_mode) if pm == 1]
                    if positions_x and positions_y:
                        min_x = min(positions_x)
                        max_x = max(positions_x)
                        min_y = min(positions_y)
                        max_y = max(positions_y)
                        input_box[0] = max(min_x - padding, input_box[0])
                        input_box[1] = max(min_y - padding, input_box[1])
                        input_box[2] = min(max_x + padding, input_box[2])
                        input_box[3] = min(max_y + padding, input_box[3])
                        input_box[:4] = [*clip_section_xyxy(
                            input_box[0],
                            input_box[1],
                            input_box[2],
                            input_box[3],
                            image
                        )]

        new_mask = segment.bbox_masks(input_box, landmarks, point_mode)

        # mask potentially overlap the bounding box, therefore use
        # padded bounding box for cutting out the mask
        new_mask = new_mask[
            0,
            int(pad_box[1]):int(pad_box[3]),
            int(pad_box[0]):int(pad_box[2])
        ]
        if new_mask.shape[0] <= 0 or new_mask.shape[1] <= 0:  # pragma: no cover # noqa: E501
            box = tracking_data.get_box(id)
            padded_box = tracking_data.get_padded_box(id)
            logging.warning(
                f'New mask is empty: {box}, {padded_box}')
        all_masks.append([new_mask])
    return SegmentationData(all_masks, down_scale)


def produce_segmentation(
    input_queue: 'Queue[DataCollection]',
    output_queue: 'Queue[DataCollection]',
    down_scale: Optional[float] = None,
    fast: bool = True,
    frame_pool: Optional[FramePool] = None,
    specific_bodypart: Optional[Synchronized] = None,
    skip_frames: bool = True,
    log_cylces: int = 100
) -> None:
    try:
        reduce_frame_discard_timer = 0.0
        timer = Timer()
        segment = MobileSam() if fast else Sam()
        frame_count = 0

        for data in pipeline_data_generator(
            input_queue,
            output_queue,
            [TrackingData],
            receiver_name='Segmentation'
        ):
            timer.tic()
            scaled_image = data.get(FrameData).get_frame(frame_pool)

            if down_scale != 1.0 and down_scale is not None:
                scaled_image = scale_image(scaled_image, 1.0 / down_scale)

            specific_bodypart_value = None if specific_bodypart is None \
                else specific_bodypart.value

            segmentation_data = segmentation_calculation(
                segment,
                scaled_image,
                down_scale,
                data.get(TrackingData),
                data.get(PoseData),
                specific_bodypart_value
            )

            if skip_frames:
                reduce_frame_discard_timer = free_output_queue(
                    output_queue, frame_pool, reduce_frame_discard_timer)
            output_queue.put(data.add(segmentation_data))

            timer.toc()
            frame_count += 1
            if frame_count == log_cylces:
                timer.clear()
            if frame_count % log_cylces == 0 and frame_count > log_cylces:
                average_time = 1. / timer.average_time, 1. / \
                    (timer.average_time + reduce_frame_discard_timer)
                logging.info(f'Segmentation-FPS: {average_time}',)
            if skip_frames and reduce_frame_discard_timer > 0.015:
                time.sleep(reduce_frame_discard_timer)
    except Exception as e:  # pragma: no cover
        if skip_frames:
            free_output_queue(output_queue, frame_pool)
        output_queue.put(DataCollection().add(
            ExceptionCloseData(e)))
    input_queue.cancel_join_thread()
    output_queue.cancel_join_thread()


class SegmentProducer:
    def __init__(
        self,
        input_queue: 'Queue[DataCollection]',
        output_queue: 'Queue[DataCollection]',
        down_scale: Optional[float] = None,
        fast: bool = True,
        frame_pool: Optional[FramePool] = None,
        specific_bodypart: Optional[Synchronized[int]] = None,
        skip_frames: bool = True
    ) -> None:
        self.process: Optional[Process] = None
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.down_scale = down_scale
        self.fast = fast
        self.frame_pool = frame_pool
        self.specific_bodypart = specific_bodypart
        self.skip_frames = skip_frames

    def start(self) -> None:
        self.process = Process(target=produce_segmentation, args=(
            self.input_queue,
            self.output_queue,
            self.down_scale,
            self.fast,
            self.frame_pool,
            self.specific_bodypart,
            self.skip_frames
        ))
        self.process.start()

    def stop(self) -> None:
        self.input_queue.put(DataCollection().add(CloseData()))
        if self.process:
            time.sleep(1)
            self.process.kill()
