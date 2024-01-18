import glob
import os
import shutil
from collections.abc import Callable
from typing import Generator

import cv2
import numpy as np
import pytest
from dotenv import load_dotenv

from frame.camera import CaptureSettings

load_dotenv()


def pytest_sessionfinish(session: pytest.Session) -> None:
    files = glob.glob('tests/tmp/*')
    for f in files:
        if os.path.isfile(f):
            os.remove(f)
        else:
            shutil.rmtree(f)


@pytest.fixture
def sample_file_2_path() -> str:
    return 'tests/resources/sample_file_2.txt'


@pytest.fixture
def sample_capture_settings() -> CaptureSettings:
    return CaptureSettings(
        input=os.path.join('tests', 'resources', 'sample_video.mp4'),
        width=1280, height=720)


@pytest.fixture
def sample_video_frame_gen() -> Callable[[], Generator[np.ndarray, None, None]]:
    def generator() -> Generator[np.ndarray, None, None]:
        cap = cv2.VideoCapture(os.path.join(
            'tests', 'resources', 'sample_video.mp4'))
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        except GeneratorExit:
            cap.release()

    return generator


def requires_env(*envs: str) -> Callable:
    env = os.environ.get('TEST_TYPE', 'slow')

    return pytest.mark.skipif(
        env not in list(envs),
        reason=f'Not suitable environment {env} for current test'
    )
