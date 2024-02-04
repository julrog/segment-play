from __future__ import annotations

import logging
import time
from multiprocessing import Process, Queue
from typing import List, Optional

import numpy as np

from frame.producer import FrameData, free_output_queue
from frame.shared import FramePool
from ocsort.timer import Timer
from pipeline.data import (BaseData, CloseData, DataCollection,
                           pipeline_data_generator)
from tracking.tracking import Tracker


class TrackingData(BaseData):
    def __init__(
        self,
        targets: List[np.ndarray],
    ) -> None:
        super().__init__()
        self.targets = targets

    def get_box(self, id: int) -> np.ndarray:
        return self.targets[id][:4].copy()

    def get_padded_box(self, id: int) -> np.ndarray:
        return self.targets[id][5:].copy()

    def get_tracking_id(self, id: int) -> int:
        return int(self.targets[id][4])


def produce_tracking(
    input_queue: Queue[DataCollection],
    output_queue: Queue[DataCollection],
    down_scale: float = 1.0,
    frame_pool: Optional[FramePool] = None,
    skip_frames: bool = True,
    log_cylces: int = 100
) -> None:
    reduce_frame_discard_timer = 0.0
    timer = Timer()
    tracker = Tracker(down_scale)
    for data in pipeline_data_generator(
        input_queue,
        output_queue,
        [FrameData]
    ):
        timer.tic()
        frame = data.get(FrameData).get_frame(frame_pool)
        tracker.update(frame)
        if skip_frames:
            reduce_frame_discard_timer = free_output_queue(
                output_queue, frame_pool, reduce_frame_discard_timer)
        output_queue.put(data.add(TrackingData(tracker.get_all_targets())))
        timer.toc()
        if tracker.current_frame == log_cylces:
            timer.clear()
        if tracker.current_frame % log_cylces == 0 and \
                tracker.current_frame > log_cylces:
            average_time = 1. / timer.average_time, 1. / \
                (timer.average_time + reduce_frame_discard_timer)
            logging.info(f'Tracking-FPS: {average_time}')
        if skip_frames and reduce_frame_discard_timer > 0.015:
            time.sleep(reduce_frame_discard_timer)
    if skip_frames:
        free_output_queue(output_queue, frame_pool)
    output_queue.cancel_join_thread()
    input_queue.cancel_join_thread()


class TrackProducer:
    def __init__(
        self,
            input_queue: Queue[DataCollection],
            output_queue: Queue[DataCollection],
            down_scale: float = 1.0,
            frame_pool: Optional[FramePool] = None,
            skip_frames: bool = True,
            log_cycles: int = 100
    ) -> None:
        self.process: Optional[Process] = None
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.down_scale = down_scale
        self.frame_pool = frame_pool
        self.skip_frames = skip_frames
        self.log_cycles = log_cycles

    def start(self) -> None:
        self.process = Process(target=produce_tracking, args=(
            self.input_queue,
            self.output_queue,
            self.down_scale,
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
