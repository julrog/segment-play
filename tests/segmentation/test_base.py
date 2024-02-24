from typing import Optional

import numpy as np
import pytest

from segmentation.base import Segmentation
from util.image import create_black_image


@pytest.fixture
def segmentation() -> Segmentation:
    return Segmentation()


def test_get_image_embedding(segmentation: Segmentation) -> None:
    image_embedding: Optional[np.ndarray] = segmentation.get_image_embedding()
    assert image_embedding is None


def test_set_image(segmentation: Segmentation) -> None:
    image: np.ndarray = create_black_image((100, 100, 3))
    with pytest.raises(AttributeError):
        segmentation.set_image(image)


def test_prepare_prompts(segmentation: Segmentation) -> None:
    image: np.ndarray = create_black_image((100, 100, 3))
    segmentation.prepare_prompts(image)
    assert True


def test_bbox_masks(segmentation: Segmentation) -> None:
    bb: np.ndarray = np.array([[0, 0, 100, 100]])
    points: np.ndarray = np.array([[50, 50]])
    point_modes: np.ndarray = np.array([1.0])
    masks: np.ndarray = segmentation.bbox_masks(bb, points, point_modes)
    assert isinstance(masks, np.ndarray)
    assert len(masks) == 0
