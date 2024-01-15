import time
from typing import Optional

import numpy as np

from frame.shared import FramePool
from pipeline.data import DataCollection


class TrackFrameStats:
    def __init__(
        self,
        frame_pool: Optional[FramePool] = None,
        history: int = 100
    ) -> None:
        self.frame_pool = frame_pool
        self.delay = np.array([-1.0 for _ in range(history)], dtype=float)
        self.processing_frames = np.array(
            [-1 for _ in range(history)], dtype=int)
        self.pointer = 0
        self.count = 0

    def add(self, frame_data: DataCollection) -> None:
        current_time = time.time()
        delay = current_time - frame_data.timestamp
        self.delay[self.pointer] = delay
        if self.frame_pool:
            self.processing_frames[self.pointer] = \
                self.frame_pool.free_frames.qsize()
        self.pointer = (self.pointer + 1) % self.delay.shape[0]
        self.count += 1

    def get_avg_delay(self) -> float:
        return np.average(self.delay[self.delay > -1.0])

    def get_processing_frames(self) -> int:
        assert self.frame_pool is not None
        filtered_frames = self.processing_frames[self.processing_frames > -1]
        if len(filtered_frames) > 0:
            return self.frame_pool.maxsize - np.average(filtered_frames)
        else:
            return 0
