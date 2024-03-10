import logging
import time
from multiprocessing import Queue, Value
from multiprocessing.sharedctypes import Synchronized
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from frame.producer import FrameData
from frame.shared import FramePool
from ocsort.timer import Timer
from pipeline.data import DataCollection, pipeline_data_generator
from pipeline.producer import Producer
from util.image import create_black_image


class Background:
    def __init__(
        self,
        frame_pool: Optional[FramePool] = None,
        new_weight: float = 0.05
    ) -> None:
        self.avg: Optional[Union[np.ndarray, int]] = None
        self.frame_pool = frame_pool
        self.new_weight = new_weight

    def add_frame(self, img: np.ndarray) -> None:
        if self.avg is None:
            if self.frame_pool is not None:
                self.avg = self.frame_pool.put(img.astype(np.float32))
            else:
                self.avg = img.astype(np.float32)
        else:
            if self.frame_pool is not None:
                cv2.accumulateWeighted(
                    img, self.frame_pool.get(self.avg), self.new_weight)
            else:
                cv2.accumulateWeighted(img, self.avg, self.new_weight)

    def add_black(self, shape: Tuple[int, ...]) -> None:
        black_image = create_black_image(shape)
        self.add_frame(black_image)

    def get_bg(self) -> np.ndarray:
        assert self.avg is not None
        if self.frame_pool is not None:
            return cv2.convertScaleAbs(self.frame_pool.get(self.avg))
        return cv2.convertScaleAbs(self.avg)

    def close(self) -> None:
        if self.avg is not None:
            if self.frame_pool is not None:
                self.frame_pool.free_frame(self.avg)
            self.avg = None


class BackgroundHandle:
    def __init__(
        self,
        background_id: 'Synchronized[int]',
        update_count: 'Synchronized[int]',
        frame_pool: FramePool,
        queue: 'Queue[DataCollection]',
    ) -> None:
        self.background_id = background_id
        self.update_count = update_count
        self.frame_pool = frame_pool
        self.queue = queue

    def add_frame(self, data: DataCollection) -> None:
        assert data.has(FrameData)
        self.queue.put(DataCollection().add(data.get(FrameData)))

    def wait_for_bg(self) -> None:
        while self.background_id.value == -1:
            time.sleep(0.01)

    def wait_for_id(self, id: int) -> None:
        while self.update_count.value < id:
            time.sleep(0.01)

    def get_bg(self) -> np.ndarray:
        assert self.background_id.value != -1
        return cv2.convertScaleAbs(
            self.frame_pool.get(self.background_id.value))


def handle_background(
    input_queue: 'Queue[DataCollection]',
    frame_pool: Optional[FramePool],
    bg_frame_pool: FramePool,
    background_id: 'Synchronized[int]',
    update_count: 'Synchronized[int]',
    log_cylces: int = 100,
) -> None:
    try:
        count = 0
        timer = Timer()
        background = Background(bg_frame_pool)
        inital_frame_set = False

        for data in pipeline_data_generator(
            input_queue,
            None,
            [FrameData],
            receiver_name='Tracking'
        ):
            timer.tic()
            frame = data.get(FrameData).get_frame(frame_pool)
            if not inital_frame_set:
                background.add_black(frame.shape)
                background.add_frame(frame)
                assert background.avg is not None
                background_id.value = background.avg
                inital_frame_set = True
            else:
                background.add_frame(frame)
            update_count.value = count

            timer.toc()
            count += 1
            if count == log_cylces:
                timer.clear()
            if count % log_cylces == 0 and count > log_cylces:
                average_time = 1. / max(timer.average_time, 0.0001)
                logging.info(f'Background-FPS: {average_time}')
    except Exception as e:  # pragma: no cover
        logging.error(f'Background exception: {e}')
    input_queue.cancel_join_thread()
    background.close()


class BackgroundProcessor(Producer):
    def __init__(
        self,
        input_queue: 'Queue[DataCollection]',
        frame_pool: Optional[FramePool],
        bg_frame_pool: FramePool,
        log_cylces: int = 100,
    ) -> None:
        self.background_id: 'Synchronized[int]' = Value(
            'i', -1)  # type: ignore
        self.update_count: 'Synchronized[int]' = Value(
            'i', -1)  # type: ignore
        super().__init__(
            input_queue,
            frame_pool,
            bg_frame_pool,
            self.background_id,
            self.update_count,
            log_cylces
        )

    def start(self, handle_logs: bool = True) -> None:
        self.base_start(handle_background, handle_logs)
