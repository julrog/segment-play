import random
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from frame.producer import FrameData
from masking.mask import (add_masks, apply_mask_grayscale, dilate,
                          dilate_single, scale_mask)
from pipeline.data import BaseData, DataCollection
from pose.analyze import detect_narrow_arms
from pose.pose import PoseRenderer
from pose.producer import PoseData
from segmentation.producer import SegmentationData
from settings import GameSettings
from tracking.producer import TrackingData
from util.image import create_black_image
from util.visualize import show_box


@dataclass
class ModificationInstance:
    tracking_id: int = -1
    visibility: bool = True
    mirror: bool = False
    extract_from_bg: bool = True
    gray_out: bool = False


class FrameModificationData(BaseData):
    def __init__(self, setting: GameSettings) -> None:
        super().__init__()
        self.setting = setting
        self.modifications: List[ModificationInstance] = []

    def add_modification(
        self,
        tracking_id: int,
        visibility: bool = True,
        mirror: bool = False,
        extract_from_bg: bool = True,
        gray_out: bool = False,
    ) -> None:
        self.modifications.append(ModificationInstance(
            tracking_id, visibility, mirror, extract_from_bg, gray_out))

    def get_tracking_id(self, id: int) -> int:
        return self.modifications[id].tracking_id


class ModifiedFrameData(FrameData):
    pass


def modification_handling(
    setting: GameSettings,
    data: DataCollection
) -> DataCollection:
    request = FrameModificationData(setting)
    tracking_data = data.get(TrackingData)
    segmentation_data = data.get(SegmentationData)
    for id in range(len(tracking_data.targets)):
        track_id = tracking_data.get_tracking_id(id)
        if track_id not in setting.id_position_map.keys():
            setting.update_position_map(track_id, random.random())

        if segmentation_data.get_mask(id) is None:
            continue

        mirror = int(track_id) % 2 == 0 and setting.get('random_people_mirror')
        visible = not detect_narrow_arms(data, id)

        gray = False
        if setting.get('gray_game'):
            image_shape = data.get(FrameData).shape
            pad_box = tracking_data.get_padded_box(id).astype(np.int32)
            input_box = tracking_data.get_box(id).astype(np.int32)
            width = pad_box[2] - pad_box[0]
            center_x = (input_box[0] + input_box[2]) / 2.0
            input_box = tracking_data.get_box(id).astype(np.int32)

            color_pos = image_shape[1] * \
                setting.id_position_map[track_id] * 0.5
            mod_x_pos = int(center_x) % int(
                image_shape[1] * 0.5)
            if abs(mod_x_pos - color_pos) > width:
                gray = True

        request.add_modification(track_id, visible, mirror, gray_out=gray)
    return data.add(request)


def get_foreground_mask(data: DataCollection) -> Optional[np.ndarray]:
    modificaton_data = data.get(FrameModificationData)
    image_shape = data.get(FrameData).shape
    tracking_data = data.get(TrackingData)
    segmentation_data = data.get(SegmentationData)

    foreground_mask = None

    for mod_id in range(len(modificaton_data.modifications)):
        data_id = tracking_data.get_index(
            modificaton_data.get_tracking_id(mod_id))
        pad_box = tracking_data.get_padded_box(data_id).astype(np.int32)

        mask = segmentation_data.get_mask(data_id)

        mask_scale = segmentation_data.mask_scale
        if foreground_mask is None:
            foreground_mask = np.zeros((
                int(image_shape[0] / mask_scale),
                int(image_shape[1] / mask_scale)
            ), dtype=bool)
        foreground_mask = add_masks(
            foreground_mask,
            mask,
            (int(pad_box[1] / mask_scale),
             int(pad_box[0] / mask_scale))
        )

    if foreground_mask is not None:
        if segmentation_data.mask_scale != 1.0:
            foreground_mask = scale_mask(
                foreground_mask, segmentation_data.mask_scale)
        foreground_mask = dilate(foreground_mask)
    return foreground_mask


def apply_segmented_object(
    data: DataCollection,
    original_image: np.ndarray,
    image: np.ndarray,
) -> np.ndarray:
    modificaton_data = data.get(FrameModificationData)
    setting = modificaton_data.setting
    tracking_data = data.get(TrackingData)
    segmentation_data = data.get(SegmentationData)

    if not setting.get('all_invisibility'):
        if setting.get('hide_background'):
            image = np.zeros(image.shape, dtype=np.uint8)
        flipped_original_image = cv2.flip(original_image, 1)
        for mod_instance in modificaton_data.modifications:
            id = tracking_data.get_index(mod_instance.tracking_id)
            mask = segmentation_data.get_mask(id)
            if mod_instance.visibility:
                if segmentation_data.mask_scale != 1.0:
                    mask = scale_mask(
                        mask, segmentation_data.mask_scale)
                    mask = dilate_single(mask)
                if not setting.get('hide_background'):
                    mask = dilate(mask)
                masking_image = original_image if not mod_instance.mirror \
                    else flipped_original_image
                gray_mask = mask if not mod_instance.mirror else np.fliplr(
                    mask)
                pad_box = tracking_data.get_padded_box(id).astype(np.int32)
                x_pos = pad_box[0]
                if mod_instance.mirror:
                    x_pos = (image.shape[1] - x_pos) - \
                        (pad_box[2] - pad_box[0])

                image = apply_mask_grayscale(
                    image,
                    masking_image,
                    gray_mask,
                    mod_instance.gray_out,
                    (pad_box[1], x_pos)
                )

    if setting.get('overall_mirror'):
        image = cv2.flip(image, 1)

    if setting.get('black'):
        image = create_black_image(image.shape)

    return image


def show_pose(
    data: DataCollection,
    pose_renderer: PoseRenderer,
    image: np.ndarray
) -> np.ndarray:
    modificaton_data = data.get(FrameModificationData)
    setting = modificaton_data.setting
    tracking_data = data.get(TrackingData)
    if setting.get('show_poses') and data.has(PoseData):
        pose_data = data.get(PoseData)
        for id in range(len(tracking_data.targets)):
            if pose_data.raw_landmarks[id]:
                input_box = tracking_data.get_padded_box(id)
                image = pose_renderer.draw(
                    image,
                    pose_data.raw_landmarks[id],
                    (int(input_box[0]), int(input_box[1])),
                    ((input_box[2] - input_box[0]) / image.shape[1],
                     (input_box[3] - input_box[1]) / image.shape[0])
                )
    return image


def show_boxes(
    data: DataCollection,
    image: np.ndarray
) -> np.ndarray:
    modificaton_data = data.get(FrameModificationData)
    setting = modificaton_data.setting
    tracking_data = data.get(TrackingData)
    if setting.get('show_boxes'):
        for id in range(len(tracking_data.targets)):
            input_box = tracking_data.get_box(id)
            track_id = tracking_data.get_tracking_id(id)
            image = show_box(image, input_box, track_id)
    return image
