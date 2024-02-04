from argparse import ArgumentParser
from typing import List, Tuple

import pytest
from conftest import requires_env
from cv2 import VideoCapture

from frame.camera import (add_camera_parameters, check_camera,
                          parse_camera_settings, set_camera_parameters)


@pytest.mark.parametrize('input', [None, 0, 1])
@pytest.mark.parametrize('width', [None, 1280, 1920])
@pytest.mark.parametrize('height', [None, 720, 1080])
@pytest.mark.parametrize('fps', [None, 30, 60])
@pytest.mark.parametrize('codec', [None, 'MJPG', 'YUV2'])
def test_arguments_and_parsing(
        input: int, width: int, height: int, fps: int, codec: str) -> None:
    arg_names = ['cam_input', 'cam_width',
                 'cam_height', 'cam_fps', 'cam_codec']
    arg_defaults = [0, 1920, 1080, 30, 'MJPG']
    arguments = []
    expected = {}
    for name, value, default in zip(
            arg_names, [input, width, height, fps, codec], arg_defaults):
        if value is not None:
            arguments.append((f'--{name}').replace('_', '-'))
            arguments.append(str(value))
            expected[name] = value
        else:
            expected[name] = default
    parser = ArgumentParser()
    parser = add_camera_parameters(parser)
    parsed = parser.parse_args(arguments)
    assert vars(parsed) == expected

    cam_settings = parse_camera_settings(vars(parsed))
    for name, value, default in zip(
            arg_names, [input, width, height, fps, codec], arg_defaults):
        if value is not None:
            assert getattr(cam_settings, name.replace('cam_', '')) == value
        else:
            assert getattr(cam_settings, name.replace('cam_', '')) == default


@pytest.mark.parametrize('arguments_pass_match', [
    ([], True),
    (['--cam-width', '2000'], False)
])
@requires_env('cam_tests')
def test_cam_check(arguments_pass_match: Tuple[List[str], bool]) -> None:
    arguments, passing = arguments_pass_match
    parser = ArgumentParser()
    parser = add_camera_parameters(parser)
    cam_settings = parse_camera_settings(vars(parser.parse_args(arguments)))
    if passing:
        cap = VideoCapture(cam_settings.input, cam_settings.api)
        set_camera_parameters(cap, cam_settings)
        check_camera(cap, cam_settings)
    else:
        with pytest.raises(AssertionError):
            cap = VideoCapture(cam_settings.input, cam_settings.api)
            set_camera_parameters(cap, cam_settings)
            check_camera(cap, cam_settings)
