from typing import Optional

import numpy as np
import pytest

from tests.ai_tester import AITester
from tests.conftest import requires_env
from util.image import create_black_image, scale_image
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
    assert expected_colors * 3 > colored_pixels_big
    assert colored_pixels_big >= expected_colors * 3 * 0.95

    cutout_image = modified_image[y1 + 2:y2 - 2, x1 + 2:x2 - 2, :]
    colored_pixels_small = (cutout_image > 0).sum()
    assert colored_pixels_small == 0


@requires_env('ai_tests')
def test_draw_box_ai(ai_tester: AITester, sample_image: np.ndarray) -> None:
    scaled_image = scale_image(sample_image, 0.1)
    x1, y1, x2, y2 = 40, 30, 60, 70
    modified_image = show_box(scaled_image.copy(), [x1, y1, x2, y2], 1)
    assert not ai_tester.binary_question(
        scaled_image, 'Is there a colored box?')
    assert ai_tester.binary_question(
        modified_image, 'Is there a colored box?')
