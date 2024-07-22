

import logging
import time
from multiprocessing import Queue, Value
from multiprocessing.sharedctypes import Synchronized
from typing import Dict, Optional, Type

from background import BackgroundHandle
from frame.producer import FrameData, free_output_queue
from frame.shared import FramePool
from masking.frame_modification import (ModifiedFrameData,
                                        apply_segmented_object,
                                        get_foreground_mask,
                                        modification_handling)
from masking.mask import apply_mask
from ocsort.timer import Timer
from pipeline.data import (DataCollection, ExceptionCloseData,
                           pipeline_data_generator)
from pipeline.producer import Producer
from segmentation.producer import SegmentationData
from settings import GameSettings
from tracking.producer import TrackingData
from util.image import create_black_image


def produce_masking(
    input_queue: 'Queue[DataCollection]',
    output_queue: 'Queue[DataCollection]',
    ready: 'Synchronized[int]',
    setting: GameSettings,
    background: BackgroundHandle,
    frame_pool: Optional[FramePool] = None,
    modified_frame_pool: Optional[FramePool] = None,
    skip_frames: bool = True,
    log_cylces: int = 100,
) -> None:
    try:
        frame_pools: Dict[Type, Optional[FramePool]] = {
            FrameData: frame_pool, ModifiedFrameData: modified_frame_pool}
        reduce_frame_discard_timer = 0.0
        timer = Timer()
        frame_count = 0
        current_bg_id: Optional[int] = None
        ready.value = 1

        for data in pipeline_data_generator(
            input_queue,
            output_queue,
            [FrameData, TrackingData, SegmentationData],
            receiver_name='Masking'
        ):
            timer.tic()
            frame = data.get(FrameData).get_frame(frame_pool)
            data = modification_handling(setting, data)

            foreground_mask = get_foreground_mask(data)

            if current_bg_id is None:
                background_image = create_black_image(frame.shape)
                current_bg_id = 0
            else:
                background.wait_for_id(current_bg_id)
                background_image = background.get_bg()
                current_bg_id += 1

            new_background = frame
            if foreground_mask is not None:
                new_background = apply_mask(
                    frame, background_image, foreground_mask)

            background.add_frame(new_background)

            modified_frame = apply_segmented_object(
                data, frame, new_background)

            if skip_frames:
                reduce_frame_discard_timer = free_output_queue(
                    output_queue, frame_pools, reduce_frame_discard_timer)
            output_queue.put(data.add(ModifiedFrameData(
                modified_frame, modified_frame_pool)))
            timer.toc()
            frame_count += 1
            if frame_count == log_cylces:
                timer.clear()
            if frame_count % log_cylces == 0 and frame_count > log_cylces:
                average_time = 1. / timer.average_time, 1. / \
                    (timer.average_time + reduce_frame_discard_timer)
                logging.info(f'Masking-FPS: {average_time}')
            if skip_frames and reduce_frame_discard_timer > 0.015:
                time.sleep(reduce_frame_discard_timer)
    except Exception as e:  # pragma: no cover
        if skip_frames:
            free_output_queue(output_queue, frame_pools)
        output_queue.put(DataCollection().add(
            ExceptionCloseData(e)))


class MaskingProducer(Producer):
    def __init__(
        self,
            input_queue: 'Queue[DataCollection]',
            output_queue: 'Queue[DataCollection]',
            setting: GameSettings,
            background: BackgroundHandle,
            frame_pool: Optional[FramePool] = None,
            modified_frame_pool: Optional[FramePool] = None,
            skip_frames: bool = True,
            log_cycles: int = 100,
    ) -> None:
        self.ready: Synchronized[int] = Value('i', 0)  # type: ignore
        super().__init__(
            input_queue,
            output_queue,
            self.ready,
            setting,
            background,
            frame_pool,
            modified_frame_pool,
            skip_frames,
            log_cycles
        )

    def start(self, handle_logs: bool = False) -> None:
        self.base_start(produce_masking, handle_logs)
