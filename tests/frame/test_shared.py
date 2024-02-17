import pickle
from multiprocessing import Process, Queue
from typing import Tuple

import numpy as np
import pytest
from conftest import requires_env

from frame.camera import CaptureSettings
from frame.shared import FramePool, create_frame_pool


@pytest.mark.parametrize('template', [np.zeros([10, 10], dtype=np.uint8)])
@pytest.mark.parametrize('maxsize', [4])
def test_frame_pool_pickle(template: np.ndarray, maxsize: int) -> None:
    frame_pool = FramePool(template, maxsize)
    assert len(frame_pool.frame_pool) == maxsize
    assert len(frame_pool.shared_memory) == maxsize
    assert frame_pool.is_empty()

    assert template.nbytes == 100
    frame_pool.free_frames = None  # ignore queue for same process pickling
    unpickeled: FramePool = pickle.loads(pickle.dumps(frame_pool))
    with pytest.raises(AttributeError):
        assert unpickeled.memory_manager is None
    assert len(unpickeled.frame_pool) is maxsize

    unpickeled_again: FramePool = pickle.loads(
        pickle.dumps(unpickeled))
    with pytest.raises(AttributeError):
        assert unpickeled_again.memory_manager is None
    assert len(unpickeled_again.frame_pool) is maxsize

    frame_pool.free_frames = Queue()
    unpickeled.free_frames = Queue()
    unpickeled_again.free_frames = Queue()


def test_full_frame_pool() -> None:
    frame_pool = FramePool(np.zeros([10, 10], dtype=np.uint8), 2)
    assert frame_pool.has_free_slots()
    frame_pool.put(np.zeros([10, 10], dtype=np.uint8))
    assert frame_pool.has_free_slots()
    frame_pool.put(np.zeros([10, 10], dtype=np.uint8))
    assert not frame_pool.has_free_slots()
    with pytest.raises(ValueError):
        frame_pool.put(np.zeros([10, 10], dtype=np.uint8))


def modify_frame(pool: FramePool, number: int) -> None:
    modify_frame = pool.get(number)
    modify_frame[5, 5] = modify_frame[5, 5] * modify_frame[5, 5]


def test_pool_multi() -> None:
    frame = np.zeros([10, 10], dtype=np.uint8)
    pool = FramePool(frame, 2)
    assert pool.free_frames.qsize() == 2
    assert pool.is_empty()

    frame[5, 5] = 2
    number = pool.put(frame)
    assert pool.free_frames.qsize() == 1

    readded_number = pool.put(number)
    assert pool.free_frames.qsize() == 1
    assert not pool.is_empty()
    assert readded_number == number

    worker_process = Process(target=modify_frame, args=(pool, number))
    worker_process.start()
    worker_process.join()

    modified_frame = pool.get(number)
    assert modified_frame[5, 5] == 2 * 2
    assert pool.free_frames.qsize() == 1

    pool.free_frame(number)
    assert pool.free_frames.qsize() == 2
    assert pool.is_empty()

    del frame, modified_frame


def test_frame_pool_from_setting(
    sample_video_data: Tuple[str, int, int]
) -> None:
    pool = create_frame_pool(2, CaptureSettings(
        sample_video_data[0], sample_video_data[1], sample_video_data[2]))
    assert pool.byte_count > 0
    assert pool.frame_pool[0].shape == (
        sample_video_data[2], sample_video_data[1], 3)


def test_frame_pool_from_non_existing_video() -> None:
    with pytest.raises(ValueError):
        create_frame_pool(
            2, CaptureSettings('does_not_exist.mp4', 1, 1))


# TODO: not use camera for this
@requires_env('cam_tests')
def test_frame_pool_from_camera() -> None:
    pool = create_frame_pool(2)
    default_cam_settings = CaptureSettings()
    assert pool.byte_count > 0
    assert pool.frame_pool[0].shape == (
        default_cam_settings.height, default_cam_settings.width, 3)

    cam_settings = CaptureSettings(width=1280, height=720)
    pool = create_frame_pool(2, cam_settings)
    assert pool.byte_count > 0
    assert pool.frame_pool[0].shape == (
        cam_settings.height, cam_settings.width, 3)
