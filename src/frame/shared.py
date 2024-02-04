import queue
from multiprocessing import Manager
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from frame.camera import CaptureSettings, set_camera_parameters


class FramePool:
    def __init__(self, template: np.ndarray, maxsize: int) -> None:
        self.maxsize = maxsize
        self.dtype = template.dtype
        self.shape = template.shape
        self.byte_count = template.nbytes
        self.queue_manager = Manager()
        self.memory_manager = SharedMemoryManager()
        self.memory_manager.start()

        self.frame_pool: List[np.ndarray] = []
        self.shared_memory: List[SharedMemory] = []
        self.free_frames = self.queue_manager.Queue(maxsize)
        for index in range(maxsize):
            self.shared_memory.append(self.memory_manager.SharedMemory(
                self.byte_count))
            self.frame_pool.append(np.frombuffer(
                self.shared_memory[index].buf,
                dtype=self.dtype
            ).reshape(self.shape))
            self.free_frames.put(index)

    def __getstate__(self) -> Dict:
        d = dict(self.__dict__)
        if 'queue_manager' in d:
            del d['queue_manager']
        if 'memory_manager' in d:
            del d['memory_manager']
        del d['frame_pool']
        return d

    def __setstate__(self, d: Dict) -> None:
        shared_memories = d['shared_memory']
        d['frame_pool'] = [
            np.frombuffer(
                sm.buf,
                dtype=d['dtype'],
                count=d['byte_count']
            ).reshape(d['shape'])
            for sm in shared_memories
        ]
        # for array in d['frame_pool']:
        #   array.flags.writeable = False
        self.__dict__.update(d)

    def free_frame(self, index: int) -> None:
        self.free_frames.put(index)

    def put(self, frame: Union[np.ndarray, int]) -> int:
        if type(frame) is not np.ndarray:
            return frame
        try:
            index: int = self.free_frames.get_nowait()
        except queue.Empty:
            raise ValueError('No free frame slots available')
        self.frame_pool[index][:] = frame[:]
        return index

    def get(self, index: int) -> np.ndarray:
        return self.frame_pool[index]

    def is_empty(self) -> bool:
        return self.free_frames.qsize() == self.maxsize

    def has_free_slots(self) -> bool:
        return self.free_frames.qsize() > 0

    def close(self) -> None:
        self.memory_manager.shutdown()


def create_frame_pool(
    maxsize: int,
    settings: Optional[CaptureSettings] = None
) -> FramePool:
    if not settings:  # pragma: cam-tests
        settings = CaptureSettings()
    if isinstance(settings.input, int):  # pragma: cam-tests
        cap = cv2.VideoCapture(settings.input, settings.api)
        set_camera_parameters(cap, settings)
    else:
        cap = cv2.VideoCapture(settings.input)

    while True:
        ret = cap.grab()
        if ret:
            ret, frame = cap.retrieve()
        if ret:  # pragma: no cover
            frame_pool = FramePool(frame, maxsize)
            cap.release()
            return frame_pool
