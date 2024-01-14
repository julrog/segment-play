import glob
import os
import shutil
from typing import Callable

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


def requires_env(*envs: str) -> Callable:
    env = os.environ.get('TEST_TYPE', 'slow')

    return pytest.mark.skipif(
        env not in list(envs),
        reason=f'Not suitable environment {env} for current test'
    )
