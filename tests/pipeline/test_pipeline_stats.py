import time
from typing import Optional

import numpy as np
import pytest

from frame.shared import FramePool
from pipeline.data import DataCollection
from pipeline.stats import TrackFrameStats


@pytest.mark.parametrize('use_frame_pool', [False, True])
def test_track_frame_stats(
        use_frame_pool: bool) -> None:
    frame_pool: Optional[FramePool] = FramePool(
        np.zeros([10, 10]), 3) if use_frame_pool else None

    stats = TrackFrameStats(frame_pool, 2)

    data = DataCollection()
    time.sleep(0.1)

    if frame_pool:
        assert stats.get_processing_frames() == 0
        frame_pool.put(np.zeros([10, 10]))

    stats.add(data)
    assert 0.2 > stats.get_avg_delay() >= 0.1
    if frame_pool:
        assert stats.get_processing_frames() == 1
        frame_pool.put(np.zeros([10, 10]))

    stats.add(DataCollection())
    assert 0.06 > stats.get_avg_delay() >= 0.05
    if frame_pool:
        assert stats.get_processing_frames() == 1.5
        frame_pool.put(np.zeros([10, 10]))

    stats.add(DataCollection())
    assert 0.01 > stats.get_avg_delay()
    if frame_pool:
        assert stats.get_processing_frames() == 2.5
