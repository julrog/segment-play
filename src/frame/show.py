from __future__ import annotations

import logging
from multiprocessing import Queue
from typing import Optional

import cv2

from frame.producer import FrameData
from frame.shared import FramePool
from ocsort.timer import Timer
from pipeline.data import DataCollection, pipeline_data_generator
from pipeline.producer import Producer

WINDOW_NAME = 'frame-window'


def produce_window(
    input_queue: 'Queue[DataCollection]',
    output_queue: 'Optional[Queue[DataCollection]]',
    key_queue: 'Queue[int]',
    frame_pool: Optional[FramePool] = None,
    log_cylces: int = 100,
    window_name: str = WINDOW_NAME,
) -> None:
    try:
        count = 0
        reduce_frame_discard_timer = 0.0
        timer = Timer()

        for data in pipeline_data_generator(
            input_queue,
            output_queue,
            [FrameData],
            receiver_name=f'Window-{window_name}'
        ):
            timer.tic()
            frame = data.get(FrameData).get_frame(frame_pool)

            cv2.imshow(WINDOW_NAME, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1) & 255
            if key != 255:
                key_queue.put(key)  # pragma: no cover

            if output_queue is not None:
                output_queue.put(data)

            timer.toc()
            count += 1

            if count == log_cylces:
                timer.clear()
            if count % log_cylces == 0 and \
                    count > log_cylces:
                average_time = 1. / timer.average_time, 1. / \
                    (timer.average_time + reduce_frame_discard_timer)
                logging.info(f'Window-{window_name}-FPS: {average_time}')
    except Exception as e:  # pragma: no cover
        logging.error(f'Window producer exception: {e}')
    input_queue.cancel_join_thread()
    if output_queue:
        output_queue.cancel_join_thread()
    key_queue.cancel_join_thread()
    cv2.destroyWindow(WINDOW_NAME)


class WindowProducer(Producer):
    def __init__(
        self,
            input_queue: 'Queue[DataCollection]',
            output_queue: 'Optional[Queue[DataCollection]]',
            key_queue: 'Queue[int]',
            frame_pool: Optional[FramePool] = None,
            log_cycles: int = 100,
            window_name: str = WINDOW_NAME,
    ) -> None:
        super().__init__(
            input_queue,
            output_queue,
            key_queue,
            frame_pool,
            log_cycles,
            window_name
        )

    def start(self) -> None:
        self.base_start(produce_window)
