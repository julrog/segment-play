import numpy as np
import pytest

from frame.producer import FrameData
from pipeline.data import DataCollection
from pose.producer import PoseData
from segmentation.sam import Sam
from tracking.producer import TrackingData


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
