

import logging
from multiprocessing import Queue
from typing import Optional

import numpy as np
import pytest

from background import (Background, BackgroundHandle, BackgroundProcessor,
                        handle_background)
from frame.producer import FrameData
from frame.shared import FramePool
from pipeline.data import CloseData, DataCollection
from pipeline.manager import clear_queue
from util.image import create_black_image


@pytest.mark.parametrize('use_frame_pool', [False, True])
@pytest.mark.parametrize('start_with_black', [False, True])
def test_background(use_frame_pool: bool, start_with_black: bool) -> None:
    weight = 0.05
    frame_pool: Optional[FramePool] = FramePool(
        create_black_image((100, 100, 3)).astype(np.float32), 2) \
        if use_frame_pool else None
    background = Background(frame_pool, weight)
    black_image = create_black_image((100, 100, 3))

    if start_with_black:
        background.add_black((100, 100, 3))
        current_background = background.get_bg()
        assert np.array_equal(current_background, black_image)

        background.add_black((100, 100, 3))
        current_background = background.get_bg()
        assert np.array_equal(current_background, black_image)
    else:
        background.add_frame(black_image)
        current_background = background.get_bg()
        assert np.array_equal(current_background, black_image)

    white_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    for i in range(4):
        background.add_frame(white_image)
        current_background = background.get_bg()
        scale = 1.0 - ((1.0 - weight) ** (i + 1))
        expected_image = white_image * scale
        expected_image = np.round(expected_image).astype(np.uint8)
        assert np.array_equal(current_background, expected_image)

    background.close()
    if frame_pool is not None:
        assert frame_pool.free_frames.qsize() == 2
    assert background.avg is None


def test_background_exception() -> None:
    background = Background()
    with pytest.raises(AssertionError):
        background.get_bg()
    background.close()


def test_background_handle() -> None:
    weight = 0.05
    # FramePool(create_black_image((100, 100, 3)), 20)
    frame_pool: Optional[FramePool] = None
    frame_pools = {FrameData: frame_pool}
    bg_frame_pool: Optional[FramePool] = FramePool(
        create_black_image((100, 100, 3)).astype(np.float32), 20)
    frame_queue: 'Queue[DataCollection]' = Queue()
    black_image = create_black_image((100, 100, 3))
    used_frames = [black_image]

    processor = BackgroundProcessor(frame_queue, None, bg_frame_pool)
    processor.start()

    handle = BackgroundHandle(processor.background_id,
                              processor.update_count,
                              bg_frame_pool, frame_queue)

    handle.add_frame(used_frames[-1])
    handle.wait_for_bg()
    current_background = handle.get_bg()
    assert np.array_equal(current_background, black_image)

    white_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    used_frames.append(white_image)
    for i in range(4):
        handle.add_frame(used_frames[-1])
        handle.wait_for_id(i + 1)
        current_background = handle.get_bg()
        scale = 1.0 - ((1.0 - weight) ** (i + 1))
        expected_image = white_image * scale
        expected_image = np.round(expected_image).astype(np.uint8)
        assert np.array_equal(current_background, expected_image)

    processor.stop()
    clear_queue(frame_queue, frame_pools)


def test_background_handle_logs(caplog: pytest.LogCaptureFixture) -> None:
    frame_pool: Optional[FramePool] = FramePool(
        create_black_image((100, 100, 3)), 20)
    frame_pools = {FrameData: frame_pool}
    bg_frame_pool: Optional[FramePool] = FramePool(
        create_black_image((100, 100, 3)).astype(np.float32), 20)
    frame_queue: 'Queue[DataCollection]' = Queue()
    black_image = create_black_image((100, 100, 3))
    used_frames = [black_image]

    processor = BackgroundProcessor(frame_queue, None, bg_frame_pool, 2)

    handle = BackgroundHandle(processor.background_id,
                              processor.update_count,
                              bg_frame_pool, frame_queue)

    with caplog.at_level(logging.INFO):
        for _ in range(4):
            handle.add_frame(used_frames[-1])
        frame_queue.put(DataCollection().add(CloseData()))

        handle_background(frame_queue, frame_pool, bg_frame_pool,
                          processor.background_id, processor.update_count, 2)

        clear_queue(frame_queue, frame_pools)

        log_tuples = caplog.record_tuples
        assert len(log_tuples) == 1
        for log_tuple in log_tuples:
            assert log_tuple[0] == 'root'
            assert log_tuple[1] == logging.INFO
            assert log_tuple[2].startswith('Background-FPS:')
