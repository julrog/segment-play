import numpy as np
import pytest

from frame.producer import FrameData
from pipeline.data import DataCollection
from pose.producer import PoseData
from segmentation.sam import Sam
from tests.ai_tester import AITester
from tests.conftest import requires_env
from tracking.producer import TrackingData
from util.image import scale_image
from util.mask import apply_mask


@pytest.fixture
def sam() -> Sam:
    return Sam()


def test_get_image_embedding(sam: Sam, sample_image: np.ndarray) -> None:
    sam.set_image(sample_image)
    image_embedding = sam.get_image_embedding()
    assert isinstance(image_embedding, np.ndarray)


def test_set_image(
        sam: Sam,
        track_data_collection: DataCollection
) -> None:
    sam.set_image(track_data_collection.get(FrameData).get_frame())
    assert isinstance(sam.image_embedding, np.ndarray)


def test_bbox_masks_no_pose(
    sam: Sam,
    track_data_collection: DataCollection
) -> None:
    bb = track_data_collection.get(TrackingData).get_box(0)
    points = None
    point_modes = None
    sam.set_image(track_data_collection.get(FrameData).get_frame())
    masks = sam.bbox_masks(bb, points, point_modes)
    assert isinstance(masks, np.ndarray)
    assert len(masks) > 0


@pytest.mark.parametrize('no_point_labels', [True, False])
def test_bbox_masks_pose(
    sam: Sam,
    pose_data_collection: DataCollection,
    no_point_labels: bool
) -> None:
    bb = pose_data_collection.get(TrackingData).get_box(0)
    points, point_mode = pose_data_collection.get(
        PoseData).get_landmarks_xy(0)
    sam.set_image(pose_data_collection.get(FrameData).get_frame())
    masks = sam.bbox_masks(
        bb,
        points,
        point_mode if no_point_labels else None
    )
    assert isinstance(masks, np.ndarray)
    assert len(masks) > 0


@requires_env('ai_tests')
def test_segmentation_ai(
    ai_tester: AITester,
    sam: Sam,
    track_data_collection: DataCollection
) -> None:
    bb = track_data_collection.get(TrackingData).get_box(0)
    points = None
    point_modes = None
    sam.set_image(track_data_collection.get(FrameData).get_frame())
    masks = sam.bbox_masks(bb, points, point_modes)

    image = track_data_collection.get(FrameData).get_frame()
    black_image = np.zeros_like(image)
    masked_image = apply_mask(black_image.copy(), image, masks[0])
    reversed_masked_image = apply_mask(image.copy(), black_image, masks[0])

    assert not ai_tester.binary_question(
        scale_image(image, 0.1), 'Is there cut out person on a black background?')  # noqa: E501
    assert ai_tester.binary_question(
        scale_image(masked_image, 0.1), 'Is there cut out person on a black background?')  # noqa: E501
    assert not ai_tester.binary_question(
        scale_image(image, 0.1), 'Is there a black silhouette in the picture?')  # noqa: E501
    assert ai_tester.binary_question(
        scale_image(reversed_masked_image, 0.1), 'Is there a black silhouette in the picture?')  # noqa: E501
