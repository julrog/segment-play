from __future__ import annotations

import logging
import time
from multiprocessing import Queue
from typing import Dict, List, Tuple, Type

from frame.producer import FrameData
from frame.shared import FramePool
from pipeline.data import DataCollection, pipeline_data_and_empty_generator
from pipeline.producer import Producer


def filter_old_frames(
    current_time: float,
    cleanup_list: List[Tuple[int, float]],
    frame_pool: FramePool,
    cleanup_delay: float = 0.0,
) -> None:
    i = 0
    while i < len(cleanup_list):
        if current_time - cleanup_list[i][1] >= cleanup_delay:
            frame_pool.free_frame(cleanup_list[i][0])
            cleanup_list.pop(i)
        else:
            i += 1


def filter_frames_limit(
    cleanup_list: List[Tuple[int, float]],
    frame_pool: FramePool,
    limit: int,
) -> None:
    while len(cleanup_list) > limit and len(cleanup_list) > 0:
        oldest_time = cleanup_list[0][1]
        oldest_index = 0
        for i in range(1, len(cleanup_list)):
            if cleanup_list[i][1] < oldest_time:
                oldest_time = cleanup_list[i][1]
                oldest_index = i
        frame_pool.free_frame(cleanup_list[oldest_index][0])
        cleanup_list.pop(oldest_index)


def clean_frame(
    input_queue: 'Queue[DataCollection]',
    frame_pools: Dict[Type, FramePool],
    cleanup_delay: float = 0.0,
    limit: int = 20,
) -> None:
    cleanup_list: Dict[Type, List[Tuple[int, float]]] = dict()
    for data_type, frame_pool in frame_pools.items():
        cleanup_list[data_type] = []
    try:
        for data in pipeline_data_and_empty_generator(
            input_queue,
            None,
            [],
            receiver_name='CleanFrame'
        ):
            current_time = time.time()
            for data_type, frame_pool in frame_pools.items():
                if data and data.has(data_type):
                    current_frame_data = data.get(data_type)
                    assert isinstance(current_frame_data, FrameData)
                    if current_frame_data.using_shared_pool:
                        cleanup_list[data_type].append(
                            (data.get(FrameData).frame, current_time))

            for data_type, frame_pool in frame_pools.items():
                filter_old_frames(current_time, cleanup_list[data_type],
                                  frame_pool, cleanup_delay)
                filter_frames_limit(cleanup_list[data_type], frame_pool, limit)

    except Exception as e:  # pragma: no cover
        logging.error(f'Frame cleanup exception: {e}')
    for data_type, frame_pool in frame_pools.items():
        for frame, _ in cleanup_list[data_type]:
            frame_pool.free_frame(frame)
    input_queue.cancel_join_thread()


class CleanFrameProducer(Producer):
    def __init__(
        self,
            input_queue: 'Queue[DataCollection]',
            frame_pools: Dict[Type, FramePool],
            cleanup_delay: float = 0.0,
            limit: int = 100,
    ) -> None:
        super().__init__(
            input_queue,
            frame_pools,
            cleanup_delay,
            limit
        )

    def start(self, handle_logs: bool = False) -> None:
        self.base_start(clean_frame, handle_logs)
