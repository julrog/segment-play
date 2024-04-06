# flake8: noqa

from __future__ import annotations

import argparse
import os.path
import sys
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(sys.modules[__name__].__file__), '..')))  # type: ignore  # noqa

import queue
from multiprocessing import Queue, freeze_support

import cv2

from frame.camera import add_camera_parameters, parse_camera_settings
from frame.clean import CleanFrameProducer
from frame.producer import FrameData, VideoCaptureProducer
from frame.shared import create_frame_pool
from frame.show import WindowProducer
from pipeline.data import DataCollection
from pipeline.manager import clear_queue
from util.logging import logger_manager


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(
        'Basic hide people via segmentation and background estimation.')
    parser = add_camera_parameters(parser)
    return vars(parser.parse_args())


def main(args: Dict) -> None:
    logger_manager.start()
    camera_settings = parse_camera_settings(args)

    frame_queue: 'Queue[DataCollection]' = Queue()
    cleanup_queue: 'Queue[DataCollection]' = Queue()
    frame_pool = create_frame_pool(100, camera_settings)
    frame_pools = {FrameData: frame_pool}

    key_queue: 'Queue[int]' = Queue()
    window = WindowProducer(frame_queue, cleanup_queue,
                            key_queue, {FrameData: frame_pool})
    cleaner = CleanFrameProducer(
        cleanup_queue, {FrameData: frame_pool}, cleanup_delay=1.0, limit=20)
    cap = VideoCaptureProducer(frame_queue, camera_settings, frame_pool)

    window.start(True)
    cleaner.start(True)
    cap.start(True)

    try:
        while True:
            try:
                data = key_queue.get(timeout=0.01)
                if chr(data & 255) == 'q':
                    break

            except queue.Empty:
                pass
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    cap.stop()
    cleaner.join()
    window.join()
    clear_queue(frame_queue, frame_pools)
    clear_queue(cleanup_queue, frame_pools)
    clear_queue(key_queue)
    logger_manager.close()
    print('Closing')


if __name__ == '__main__':
    freeze_support()
    main(parse_args())
