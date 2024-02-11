import numpy as np
import pytest

from frame.producer import FrameData
from pipeline.data import DataCollection
from pose.producer import PoseData
from segmentation.mobile_sam import MobileSam
from tracking.producer import TrackingData


@pytest.fixture
def mobile_sam() -> MobileSam:
    return MobileSam()


def test_get_image_embedding(
    mobile_sam: MobileSam,
    sample_image: np.ndarray
) -> None:
    mobile_sam.set_image(sample_image)
    image_embedding = mobile_sam.get_image_embedding()
    assert isinstance(image_embedding, np.ndarray)


def test_set_image(
        mobile_sam: MobileSam,
        track_data_collection: DataCollection
) -> None:
    mobile_sam.set_image(track_data_collection.get(FrameData).get_frame())
    assert isinstance(mobile_sam.image_embedding, np.ndarray)


def test_bbox_masks_no_pose(
    mobile_sam: MobileSam,
    track_data_collection: DataCollection
) -> None:
    bb = track_data_collection.get(TrackingData).get_box(0)
    points = None
    point_modes = None
    mobile_sam.set_image(track_data_collection.get(FrameData).get_frame())
    masks = mobile_sam.bbox_masks(bb, points, point_modes)
    assert isinstance(masks, np.ndarray)
    assert len(masks) > 0


@pytest.mark.parametrize('no_point_labels', [True, False])
def test_bbox_masks_pose(
    mobile_sam: MobileSam,
    pose_data_collection: DataCollection,
    no_point_labels: bool
) -> None:
    bb = pose_data_collection.get(TrackingData).get_box(0)
    points, point_mode = pose_data_collection.get(
        PoseData).get_landmarks_xy(0)
    mobile_sam.set_image(pose_data_collection.get(FrameData).get_frame())
    masks = mobile_sam.bbox_masks(
        bb,
        points,
        point_mode if no_point_labels else None
    )
    assert isinstance(masks, np.ndarray)
    assert len(masks) > 0
