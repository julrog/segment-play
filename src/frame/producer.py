from __future__ import annotations

import logging
import queue
import time
from multiprocessing import Process, Queue, Value
from multiprocessing.sharedctypes import Synchronized
from typing import Dict, Optional, Type, Union

import cv2
import numpy as np

from frame.camera import CaptureSettings, check_camera, set_camera_parameters
from frame.shared import FramePool
from pipeline.data import (BaseData, CloseData, DataCollection,
                           ExceptionCloseData)
from pipeline.producer import interruptible
from util.logging import logger_manager, logging_process


class FrameData(BaseData):
    def __init__(
        self,
        frame: Union[np.ndarray, int],
        frame_pool: Optional[FramePool] = None,
        id: Optional[int] = None
    ) -> None:
        super().__init__()
        self.shape = None if type(frame) is not np.ndarray else frame.shape
        self.id = id
        if frame_pool:
            self.using_shared_pool = True
            self.frame = frame_pool.put(frame)
        else:
            self.using_shared_pool = False
            self.frame = frame

    def get_frame(self, frame_pool: Optional[FramePool] = None) -> np.ndarray:
        if frame_pool and type(self.frame) is not np.ndarray:
            return frame_pool.get(self.frame)
        else:
            return self.frame


def free_output_queue(
        output_queue: 'Queue[DataCollection]',
        frame_pools: Dict[Type, Optional[FramePool]] = {},
        reduce_frame_discard_timer: Optional[float] = None
) -> Optional[float]:
    if not output_queue.empty():
        try:
            discarded_frame = output_queue.get(timeout=0.01)
            for data_type, frame_pool in frame_pools.items():
                if discarded_frame.has(data_type) and frame_pool:
                    data = discarded_frame.get(data_type)
                    assert isinstance(data, FrameData)
                    frame_pool.free_frame(data.frame)
            if reduce_frame_discard_timer is not None:
                reduce_frame_discard_timer += 0.015
        except queue.Empty:  # pragma: no cover
            pass
    else:
        if reduce_frame_discard_timer is not None:
            reduce_frame_discard_timer -= 0.001
            if reduce_frame_discard_timer < 0.0:
                reduce_frame_discard_timer = 0.0
    return reduce_frame_discard_timer


def produce_capture(
        output_queue: 'Queue[DataCollection]',
        settings: Optional[CaptureSettings],
        stop_condition: 'Synchronized[int]',
        frame_pool: Optional[FramePool] = None,
        skip_frames: bool = True,
) -> None:
    frame_pools: Dict[Type, Optional[FramePool]] = {FrameData: frame_pool}
    if settings:
        if isinstance(settings.input, int):  # pragma: cam-tests
            cap = cv2.VideoCapture(settings.input, settings.api)
            set_camera_parameters(cap, settings)
            check_camera(cap, settings)
        else:
            cap = cv2.VideoCapture(settings.input)
    else:
        cap = cv2.VideoCapture(0, CaptureSettings().api)  # pragma: cam-tests
    logging.info(f'Camera-FPS: {int(cap.get(cv2.CAP_PROP_FPS))}')

    try:
        count = 0
        while True:
            # grab first, otherwise the process might close unexpected with
            # read
            ret = cap.grab()
            if not ret or stop_condition.value:  # pragma: no cover
                break
            ret, frame = cap.retrieve()
            if not ret or stop_condition.value:  # pragma: no cover
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            if skip_frames:
                free_output_queue(output_queue, frame_pools)
                output_queue.put(DataCollection().add(
                    FrameData(frame, frame_pool, count)))
            else:
                placed_frame = False
                # try and wait until there is space in the queue
                while not placed_frame:
                    try:
                        output_queue.put(DataCollection().add(
                            FrameData(frame, frame_pool, count)))
                        placed_frame = True
                    except ValueError:
                        time.sleep(0.01)

            count += 1

        if skip_frames:
            free_output_queue(output_queue, frame_pools)
        output_queue.put(DataCollection().add(CloseData()))
    except Exception as e:  # pragma: no cover
        if skip_frames:
            free_output_queue(output_queue, frame_pools)
        output_queue.put(DataCollection().add(
            ExceptionCloseData(e)))
    output_queue.cancel_join_thread()
    cap.release()


class VideoCaptureProducer:
    def __init__(
        self,
        frame_queue: 'Queue[DataCollection]',
        settings: Optional[CaptureSettings] = None,
        frame_pool: Optional[FramePool] = None,
        skip_frames: bool = True
    ) -> None:
        self.settings = settings
        self.frame_queue = frame_queue
        self.process: Optional[Process] = None
        self.stop_condition: Synchronized[int] = Value('i', 0)  # type: ignore
        self.frame_pool = frame_pool
        self.skip_frames = skip_frames

    def start(self, handle_logs: bool = True) -> None:
        if handle_logs:
            self.process = Process(
                target=logging_process(interruptible), args=(
                    produce_capture,
                    self.frame_queue,
                    self.settings,
                    self.stop_condition,
                    self.frame_pool,
                    self.skip_frames
                ), kwargs=dict(logger=logger_manager.create_logger()))
        else:
            self.process = Process(target=interruptible, args=(
                produce_capture,
                self.frame_queue,
                self.settings,
                self.stop_condition,
                self.frame_pool,
                self.skip_frames
            ))
        self.process.start()

    def stop(self) -> None:
        self.stop_condition.value = 1
        if self.process:
            self.process.join(1.0)
