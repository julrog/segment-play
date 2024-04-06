from typing import Callable, Dict, Tuple

import cv2
import pytest

from background import Background
from frame.producer import FrameData
from masking.frame_modification import (FrameModificationData,
                                        apply_segmented_object,
                                        get_foreground_mask,
                                        modification_handling, show_boxes,
                                        show_pose)
from masking.mask import apply_mask
from pipeline.data import DataCollection
from pose.pose import PoseRenderer
from pose.producer import PoseData
from segmentation.producer import SegmentationData
from settings import GameSettings


def test_FrameModificationData() -> None:
    setting = GameSettings()
    frame_modification_data = FrameModificationData(setting)

    frame_modification_data.add_modification(1, False, True, False, True)
    frame_modification_data.add_modification(2)
    assert len(frame_modification_data.modifications) == 2
    assert frame_modification_data.get_tracking_id(0) == 1
    assert frame_modification_data.get_tracking_id(1) == 2

    instance_one = frame_modification_data.modifications[0]
    instance_two = frame_modification_data.modifications[1]
    for attr in instance_one.__dict__:
        assert getattr(instance_one, attr) != getattr(instance_two, attr)


@pytest.mark.parametrize('gray', [True, False])
@pytest.mark.parametrize('no_mask', [True, False])
@pytest.mark.parametrize('gray_pos_modified', [True, False])
def test_modification_handling(
    segmentation_data_collection: DataCollection,
    gray: bool,
    no_mask: bool,
    gray_pos_modified: bool
) -> None:
    setting = GameSettings()
    setting.set('gray_game', gray)
    if no_mask:
        segmentation_data_collection.get(SegmentationData).masks = []
    else:
        if gray_pos_modified:
            setting.update_position_map(1, 100000.0)

    result = modification_handling(setting, segmentation_data_collection)
    assert isinstance(result, DataCollection)
    assert result.has(FrameModificationData)


@pytest.mark.parametrize('pose_data_type', [
    'No data', 'No landmarks', 'full data'])
def test_show_pose(
    full_data_collection: DataCollection,
    segmentation_data_collection: DataCollection,
    pose_data_type: str
) -> None:
    setting = GameSettings()
    render = PoseRenderer()

    setting.set('show_poses', True)
    data = full_data_collection if pose_data_type != 'No data' \
        else segmentation_data_collection
    data = modification_handling(setting, data)
    if pose_data_type == 'No landmarks':
        data.get(PoseData).raw_landmarks[0] = None

    frame = data.get(FrameData).frame
    image = show_pose(data, render, frame.copy())

    if pose_data_type == 'full data':
        assert not (frame == image).all()
    else:
        assert (frame == image).all()


@pytest.mark.parametrize('show_boxes_setting', [True, False])
def test_show_box(
    full_data_collection: DataCollection,
    show_boxes_setting: bool
) -> None:
    setting = GameSettings()

    setting.set('show_boxes', show_boxes_setting)
    data = modification_handling(setting, full_data_collection)

    frame = data.get(FrameData).frame
    image = show_boxes(data, frame.copy())

    if show_boxes_setting:
        assert not (frame == image).all()
    else:
        assert (frame == image).all()


@pytest.mark.parametrize('no_mask', [True, False])
@pytest.mark.parametrize('down_scale', [1.0, 2.0])
def test_get_foreground_mask(
    full_data_generator: Callable[[str, float], DataCollection],
    sample_image_multiple_path: str,
    no_mask: bool,
    down_scale: float
) -> None:
    full_data_collection = full_data_generator(
        sample_image_multiple_path, down_scale)
    setting = GameSettings()
    if no_mask:
        full_data_collection.get(SegmentationData).masks = []
    data = modification_handling(setting, full_data_collection)
    foreground_mask = get_foreground_mask(data)
    if no_mask:
        assert foreground_mask is None
    else:
        assert foreground_mask is not None


@pytest.mark.parametrize('test_settings', [
    ('only_background', {'all_invisibility': True}),
    ('black_background', {'all_invisibility': False, 'hide_background': True}),
    ('original', {'all_invisibility': False, 'overall_mirror': False}),
    ('just_black', {'black': True}),
    ('default_mirrored', {})
])
def test_apply_segmented_object(
    full_data_generator: Callable[[str, float], DataCollection],
    sample_image_multiple_path: str,
    test_settings: Tuple[str, Dict[str, bool]],
) -> None:
    background: Background = Background()
    full_data_collection = full_data_generator(
        sample_image_multiple_path, 1.0)
    image = full_data_collection.get(FrameData).get_frame().copy()
    setting = GameSettings()
    setting_name, overwrite_setting = test_settings
    for key, value in overwrite_setting.items():
        setting.set(key, value)
    data = modification_handling(setting, full_data_collection)

    foreground_mask = get_foreground_mask(data)
    assert foreground_mask is not None

    background.add_black(image.shape)
    image = apply_mask(image, background.get_bg(), foreground_mask)
    background.add_frame(image)
    image = apply_segmented_object(data, full_data_collection.get(
        FrameData).get_frame(), image)

    if setting_name == 'just_black':
        assert (image == 0).all()
    else:
        assert not (image == 0).all()

    if setting_name == 'original':
        assert (image == full_data_collection.get(
            FrameData).get_frame()).all()
    elif setting_name == 'default_mirrored':
        assert (cv2.flip(image, 1) == full_data_collection.get(
            FrameData).get_frame()).all()
    else:
        assert not (image == full_data_collection.get(
            FrameData).get_frame()).all()


@pytest.mark.parametrize('test_settings', [
    ('original', {'all_invisibility': False, 'overall_mirror': False}),
])
@pytest.mark.parametrize('down_scale', [2.0])
def test_apply_segmented_object_scale(
    full_data_generator: Callable[[str, float], DataCollection],
    sample_image_multiple_path: str,
    test_settings: Tuple[str, Dict[str, bool]],
    down_scale: float
) -> None:
    background: Background = Background()
    full_data_collection = full_data_generator(
        sample_image_multiple_path, down_scale)
    image = full_data_collection.get(FrameData).get_frame().copy()
    setting = GameSettings()
    _, overwrite_setting = test_settings
    for key, value in overwrite_setting.items():
        setting.set(key, value)
    data = modification_handling(setting, full_data_collection)

    foreground_mask = get_foreground_mask(data)
    assert foreground_mask is not None

    background.add_black(image.shape)
    image = apply_mask(image, background.get_bg(), foreground_mask)
    background.add_frame(image)
    image = apply_segmented_object(data, full_data_collection.get(
        FrameData).get_frame(), image)

    assert (image == full_data_collection.get(FrameData).get_frame()).all()


@pytest.mark.parametrize('test_settings', [
    ('original', {'all_invisibility': False, 'overall_mirror': False}),
])
@pytest.mark.parametrize('action', ['mirrored', 'invisible'])
def test_apply_segmented_individual(
    full_data_generator: Callable[[str, float], DataCollection],
    sample_image_multiple_path: str,
    test_settings: Tuple[str, Dict[str, bool]],
    action: str
) -> None:
    background: Background = Background()
    full_data_collection = full_data_generator(
        sample_image_multiple_path, 1.0)
    image = full_data_collection.get(FrameData).get_frame().copy()
    setting = GameSettings()
    _, overwrite_setting = test_settings
    for key, value in overwrite_setting.items():
        setting.set(key, value)
    data = modification_handling(setting, full_data_collection)

    data.get(FrameModificationData).modifications[2].visibility = (
        action != 'invisible')
    data.get(FrameModificationData).modifications[2].mirror = (
        action == 'mirrored')

    foreground_mask = get_foreground_mask(data)
    assert foreground_mask is not None

    background.add_black(image.shape)
    image = apply_mask(image, background.get_bg(), foreground_mask)
    background.add_frame(image)
    image = apply_segmented_object(data, full_data_collection.get(
        FrameData).get_frame(), image)
    assert not (image == full_data_collection.get(FrameData).get_frame()).all()
