from __future__ import annotations

import queue
from multiprocessing import Process, Queue, Value
from multiprocessing.sharedctypes import Synchronized
from typing import Optional, Union

import cv2
import numpy as np

from frame.camera import CaptureSettings, check_camera, set_camera_parameters
from frame.shared import FramePool
from pipeline.data import BaseData, CloseData, DataCollection
from pipeline.producer import interruptible


class FrameData(BaseData):
    def __init__(
        self,
        frame: Union[np.ndarray, int],
        frame_pool: Optional[FramePool] = None
    ) -> None:
        super().__init__()
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
        output_queue: Queue[DataCollection],
        frame_pool: Optional[FramePool] = None,
        reduce_frame_discard_timer: Optional[float] = None
) -> Optional[float]:
    if not output_queue.empty():
        try:
            discarded_frame = output_queue.get(timeout=0.01)
            if frame_pool and discarded_frame.has(FrameData):
                frame_pool.free_frame(discarded_frame.get(FrameData).frame)
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
        output_queue: Queue[DataCollection],
        settings: Optional[CaptureSettings],
        stop_condition: Synchronized,
        frame_pool: Optional[FramePool] = None,
) -> None:
    if settings:
        if isinstance(settings.input, int):
            cap = cv2.VideoCapture(settings.input, settings.api)
            set_camera_parameters(cap, settings)
            check_camera(cap, settings)
        else:
            cap = cv2.VideoCapture(settings.input)
    else:
        cap = cv2.VideoCapture(0, CaptureSettings().api)
    print('Camera-FPS: ', int(cap.get(cv2.CAP_PROP_FPS)))

    while True:
        # grab first, otherwise the process might close unexpected with read
        ret = cap.grab()
        if not ret or stop_condition.value:  # pragma: no cover
            break
        ret, frame = cap.retrieve()
        if not ret or stop_condition.value:  # pragma: no cover
            break
        frame.flags.writeable = False
        free_output_queue(output_queue, frame_pool)
        output_queue.put(DataCollection().add(
            FrameData(frame, frame_pool)))

    free_output_queue(output_queue, frame_pool)
    output_queue.put(DataCollection().add(CloseData()))
    output_queue.cancel_join_thread()
    cap.release()


class VideoCaptureProducer:
    def __init__(
        self,
        frame_queue: Queue[DataCollection],
        settings: Optional[CaptureSettings] = None,
        frame_pool: Optional[FramePool] = None
    ) -> None:
        self.settings = settings
        self.frame_queue = frame_queue
        self.process: Optional[Process] = None
        self.stop_condition: Synchronized[int] = Value('i', 0)  # type: ignore
        self.frame_pool = frame_pool

    def start(self) -> None:
        self.process = Process(target=interruptible, args=(
            produce_capture,
            self.frame_queue,
            self.settings,
            self.stop_condition,
            self.frame_pool
        ))
        self.process.start()

    def stop(self) -> None:
        self.stop_condition.value = 1
        if self.process:
            self.process.join(1.0)
