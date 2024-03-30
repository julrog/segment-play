

import logging
import time
from multiprocessing import Queue
from typing import Optional

from frame.producer import FrameData, free_output_queue
from frame.shared import FramePool
from ocsort.timer import Timer
from pipeline.data import (DataCollection, ExceptionCloseData,
                           pipeline_data_generator)
from pipeline.producer import Producer
from settings import GameSettings


def produce_masking(
    input_queue: 'Queue[DataCollection]',
    output_queue: 'Queue[DataCollection]',
    setting: GameSettings,
    frame_pool: Optional[FramePool] = None,
    skip_frames: bool = True,
    log_cylces: int = 100,
) -> None:
    try:
        reduce_frame_discard_timer = 0.0
        timer = Timer()
        frame_count = 0

        for data in pipeline_data_generator(
            input_queue,
            output_queue,
            [FrameData],
            receiver_name='Masking'
        ):
            timer.tic()
            # frame = data.get(FrameData).get_frame(frame_pool)

            if skip_frames:
                reduce_frame_discard_timer = free_output_queue(
                    output_queue, frame_pool, reduce_frame_discard_timer)
            # output_queue.put(data.add())
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
            free_output_queue(output_queue, frame_pool)
        output_queue.put(DataCollection().add(
            ExceptionCloseData(e)))
    output_queue.cancel_join_thread()
    input_queue.cancel_join_thread()


class MaskingProducer(Producer):
    def __init__(
        self,
            input_queue: 'Queue[DataCollection]',
            output_queue: 'Queue[DataCollection]',
            frame_pool: Optional[FramePool] = None,
            skip_frames: bool = True,
            log_cycles: int = 100,
    ) -> None:
        super().__init__(
            input_queue,
            output_queue,
            frame_pool,
            skip_frames,
            log_cycles
        )

    def start(self, handle_logs: bool = True) -> None:
        self.base_start(produce_masking, handle_logs)
