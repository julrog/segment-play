from typing import Optional

import pytest

from util.image import create_black_image
from util.visualize import show_box


@pytest.mark.parametrize('color_id', [None, 1])
def test_draw_box(color_id: Optional[int]) -> None:
    black_image = create_black_image([100, 100, 3])
    black_image_copy = black_image.copy()

    x1, y1, x2, y2 = 20, 30, 60, 70

    modified_image = show_box(black_image_copy, [x1, y1, x2, y2], color_id)
    assert (black_image_copy == modified_image).all()
    assert not (black_image == modified_image).all()

    expected_colors = (x2 - x1) * 2 * 3 + (y2 - y1) * 2 * 3
    colored_pixels_big = (modified_image > 0).sum()
    assert expected_colors * 3 > colored_pixels_big >= expected_colors * 3 * 0.95

    cutout_image = modified_image[y1 + 2:y2 - 2, x1 + 2:x2 - 2, :]
    colored_pixels_small = (cutout_image > 0).sum()
    assert colored_pixels_small == 0
