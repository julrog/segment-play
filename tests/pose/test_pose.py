import numpy as np
import pytest

from pose.pose import Pose, PoseRenderer
from tests.ai_tester import AITester
from tests.conftest import requires_env
from util.image import create_black_image, scale_image


@pytest.fixture
def pose() -> Pose:
    return Pose()


def test_predict_raw(pose: Pose, sample_image: np.ndarray) -> None:
    result = pose.predict_raw(sample_image)
    assert result is not None
    landmarks = result.pose_landmarks
    assert landmarks is not None
    assert len(list(landmarks.landmark)) == 33
    landmark = landmarks.landmark[0]
    assert landmark.x >= -1.0 and landmark.x <= 1.0
    assert landmark.y >= -1.0 and landmark.y <= 1.0
    assert landmark.z >= -1.0 and landmark.z <= 1.0
    assert landmark.visibility >= 0.0 and landmark.visibility <= 1.0


def test_predict(pose: Pose, sample_image: np.ndarray) -> None:
    important_landmarks, raw_landmarks = pose.predict(sample_image)
    assert len(important_landmarks) == len(pose.important_landmarks)
    assert len(list(raw_landmarks.landmark)) == 33


def test_pose_render(pose: Pose, sample_image: np.ndarray) -> None:
    render = PoseRenderer()
    important_landmarks, raw_landmarks = pose.predict(sample_image)
    assert sample_image.shape == render.draw(sample_image, raw_landmarks).shape


@requires_env('ai_tests')
def test_pose_render_ai(
    ai_tester: AITester,
    pose: Pose,
    sample_image: np.ndarray
) -> None:
    render = PoseRenderer()
    scaled_image = scale_image(sample_image, 0.1)
    important_landmarks, raw_landmarks = pose.predict(scaled_image)
    rendered_pose = render.draw(scaled_image.copy(), raw_landmarks)
    assert not ai_tester.binary_question(
        scaled_image, 'Are there pose landmarks drawn?')
    assert ai_tester.binary_question(
        rendered_pose, 'Are there pose landmarks drawn?')


def test_combine_landmarks(pose: Pose, sample_image: np.ndarray) -> None:
    important_landmarks, raw_landmarks = pose.predict(sample_image)
    landmark_map = [(1.0, 0), (1.0, 11), (1.0, 13)]
    result = pose.combine_landmarks(sample_image, landmark_map, raw_landmarks)
    assert isinstance(result, tuple)
    assert len(result) == 4
    assert result[0] >= 0.0 and result[0] <= sample_image.shape[1]
    assert result[1] >= 0.0 and result[1] <= sample_image.shape[0]
    assert result[2] >= -1.0 and result[2] <= 1.0
    assert result[3] >= 0.0 and result[3] <= 1.0


def test_get_landmark(pose: Pose, sample_image: np.ndarray) -> None:
    important_landmarks, raw_landmarks = pose.predict(sample_image)
    result = pose.get_landmark(sample_image, 0, raw_landmarks)
    assert isinstance(result, tuple)
    assert len(result) == 4
    assert result[0] >= 0.0 and result[0] <= sample_image.shape[1]
    assert result[1] >= 0.0 and result[1] <= sample_image.shape[0]
    assert result[2] >= -1.0 and result[2] <= 1.0
    assert result[3] >= 0.0 and result[3] <= 1.0


def test_close() -> None:
    pose = Pose()
    pose.close()
    with pytest.raises(AssertionError):
        pose.predict(np.zeros((100, 100, 3), dtype=np.uint8))


def test_no_detection(pose: Pose) -> None:
    image = create_black_image((100, 100, 3))
    important_landmarks, raw_landmarks = pose.predict(image)
    assert len(important_landmarks) == 0
    assert raw_landmarks is None
