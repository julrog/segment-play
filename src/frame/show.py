from __future__ import annotations

import logging
from multiprocessing import Queue
from typing import Dict, Optional, Type

import cv2

from frame.producer import FrameData
from frame.shared import FramePool
from ocsort.timer import Timer
from pipeline.data import DataCollection, pipeline_data_and_empty_generator
from pipeline.producer import Producer

WINDOW_NAME = 'frame-window'


def produce_window(
    input_queue: 'Queue[DataCollection]',
    output_queue: 'Optional[Queue[DataCollection]]',
    key_queue: 'Queue[int]',
    frame_pools: Dict[Type, FramePool] = {FrameData: None},
    show_type: Type = FrameData,
    log_cylces: int = 100,
    window_name: str = WINDOW_NAME,
) -> None:
    resized = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        count = 0
        reduce_frame_discard_timer = 0.0
        timer = Timer()

        for data in pipeline_data_and_empty_generator(
            input_queue,
            output_queue,
            [FrameData],
            receiver_name=f'Window-{window_name}'
        ):
            if data:
                timer.tic()
                show_data = data.get(show_type)
                assert isinstance(show_data, FrameData)
                frame = show_data.get_frame(frame_pools[show_type])

                cv2.imshow(WINDOW_NAME, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if not resized:
                    cv2.resizeWindow(
                        WINDOW_NAME, frame.shape[1], frame.shape[0])
                    resized = True

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
            else:
                key = cv2.waitKey(1) & 255
                if key != 255:
                    key_queue.put(key)  # pragma: no cover
    except Exception as e:  # pragma: no cover
        logging.error(f'Window producer exception: {e}')
    key_queue.cancel_join_thread()
    cv2.destroyWindow(WINDOW_NAME)


class WindowProducer(Producer):
    def __init__(
        self,
            input_queue: 'Queue[DataCollection]',
            output_queue: 'Optional[Queue[DataCollection]]',
            key_queue: 'Queue[int]',
            frame_pools: Dict[Type, FramePool] = {FrameData: None},
            show_type: Type = FrameData,
            log_cycles: int = 100,
            window_name: str = WINDOW_NAME,
    ) -> None:
        super().__init__(
            input_queue,
            output_queue,
            key_queue,
            frame_pools,
            show_type,
            log_cycles,
            window_name
        )

    def start(self, handle_logs: bool = False) -> None:
        self.base_start(produce_window, handle_logs)
