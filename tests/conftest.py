import glob
import os
import shutil
from collections.abc import Callable
from typing import Generator, Tuple

import cv2
import numpy as np
import pytest
from coverage_conditional_plugin import get_env_info
from dotenv import load_dotenv

from frame.camera import CaptureSettings
from frame.producer import FrameData
from pipeline.data import DataCollection
from pose.pose import Pose
from pose.producer import region_pose_estimation
from segmentation.mobile_sam import MobileSam
from segmentation.producer import segmentation_calculation
from tests.ai_tester import AITester
from tracking.producer import TrackingData
from tracking.tracking import Tracker

load_dotenv()
get_env_info()


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
def sample_video_data() -> Tuple[str, int, int]:
    return os.path.join('tests', 'resources', 'sample_video.mp4'), 1280, 720


@pytest.fixture
def sample_video_path() -> str:
    return os.path.join('tests', 'resources', 'sample_video.mp4')


@pytest.fixture
def sample_capture_settings() -> CaptureSettings:
    return CaptureSettings(
        input=os.path.join('tests', 'resources', 'sample_video.mp4'),
        width=1280, height=720)


@pytest.fixture
def short_sample_capture_settings() -> CaptureSettings:
    return CaptureSettings(
        input=os.path.join('tests', 'resources', 'short_sample_video.mp4'),
        width=1280, height=720)


@pytest.fixture
def sample_image_path() -> str:
    return 'tests/resources/sample_image_1.jpg'


@pytest.fixture
def sample_image_2_path() -> str:
    return 'tests/resources/sample_image_2.png'


@pytest.fixture
def sample_image_multiple_path() -> str:
    return 'tests/resources/sample_image_multiple.jpg'


@pytest.fixture
def sample_image(sample_image_path: str) -> np.ndarray:
    image = cv2.imread(sample_image_path)
    return image


@pytest.fixture
def ai_tester() -> AITester:
    return AITester()


@pytest.fixture
def sample_video_frame_count() -> int:
    return 382


@pytest.fixture
def short_sample_video_frame_count() -> int:
    return 382


@pytest.fixture
def sample_video_frame_gen() -> Callable[
        [], Generator[np.ndarray, None, None]]:
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


@pytest.fixture
def track_data_collection(sample_image: np.ndarray) -> DataCollection:
    tracker = Tracker()
    tracker.update(sample_image)
    data = DataCollection()
    data.add(FrameData(sample_image))
    data.add(TrackingData(tracker.get_all_targets()))
    return data


@pytest.fixture
def pose_data_collection(sample_image: np.ndarray) -> DataCollection:
    tracker = Tracker()
    tracker.update(sample_image)
    pose = Pose()
    data = DataCollection()
    data.add(FrameData(sample_image))
    data.add(TrackingData(tracker.get_all_targets()))
    data.add(region_pose_estimation(
        pose, data.get(TrackingData), sample_image))
    return data


@pytest.fixture
def pose_data_generator() -> Callable[[str], DataCollection]:
    def generate_pose_data(image_path: str) -> DataCollection:
        image = cv2.imread(image_path)
        tracker = Tracker()
        tracker.update(image)
        pose = Pose()
        data = DataCollection()
        data.add(FrameData(image))
        data.add(TrackingData(tracker.get_all_targets()))
        data.add(region_pose_estimation(
            pose, data.get(TrackingData), image))
        return data
    return generate_pose_data


@pytest.fixture
def segmentation_data_collection(sample_image: np.ndarray) -> DataCollection:
    segmentation = MobileSam()
    tracker = Tracker()
    tracker.update(sample_image)
    data = DataCollection()
    data.add(FrameData(sample_image))
    data.add(TrackingData(tracker.get_all_targets()))
    segmentation_data = segmentation_calculation(
        segmentation, sample_image, 1.0, data.get(TrackingData), None, None
    )
    data.add(segmentation_data)
    return data


@pytest.fixture
def full_data_collection(sample_image: np.ndarray) -> DataCollection:
    segmentation = MobileSam()
    tracker = Tracker()
    pose = Pose()
    tracker.update(sample_image)
    data = DataCollection()
    data.add(FrameData(sample_image))
    data.add(TrackingData(tracker.get_all_targets()))
    data.add(region_pose_estimation(
        pose, data.get(TrackingData), sample_image))
    segmentation_data = segmentation_calculation(
        segmentation, sample_image, 1.0, data.get(TrackingData), None, None
    )
    data.add(segmentation_data)
    return data


@pytest.fixture
def full_data_generator() -> Callable[[str, float], DataCollection]:
    def generate_full_data(
        image_path: str,
        down_scale: float = 1.0
    ) -> DataCollection:
        image = cv2.imread(image_path)
        segmentation = MobileSam()
        tracker = Tracker()
        pose = Pose()
        tracker.update(image)
        data = DataCollection()
        data.add(FrameData(image))
        data.add(TrackingData(tracker.get_all_targets()))
        data.add(region_pose_estimation(
            pose, data.get(TrackingData), image))
        segmentation_data = segmentation_calculation(
            segmentation, image, down_scale, data.get(TrackingData), None, None
        )
        data.add(segmentation_data)
        return data
    return generate_full_data


def requires_env(*required_envs: str) -> Callable:
    envs = []
    if os.environ.get('CAM_TESTS', 'False') == 'True':
        envs.append('cam_tests')
    if os.environ.get('AI_TESTS', 'False') == 'True':
        envs.append('ai_tests')
    all_envs_set = True
    for required_env in required_envs:
        if required_env not in envs:
            all_envs_set = False
            break
    return pytest.mark.skipif(
        not all_envs_set,
        reason='No suitable environment for current test'
    )
