import queue
from multiprocessing import Queue
from multiprocessing.sharedctypes import Synchronized
from typing import Dict, Generator, List, Optional, Type

from frame.camera import CaptureSettings
from frame.producer import FrameData, VideoCaptureProducer
from frame.shared import FramePool
from pipeline.data import DataCollection
from pose.producer import PoseProducer
from segmentation.producer import SegmentProducer
from tracking.producer import TrackProducer


class FrameProcessingPipeline:
    def __init__(
        self,
        segment_processes: int = 2,
        down_scale: float = 1.0,
        fast: bool = True,
        camera_settings: Optional[CaptureSettings] = None,
        frame_pool: Optional[FramePool] = None,
        specific_bodypart: Optional[Synchronized] = None,
        use_pose: bool = False,
        skip_capture_frames: bool = True
    ) -> None:
        self.frame_pools = {FrameData: frame_pool}
        self.use_pose = use_pose

        self.frame_queue: 'Queue[DataCollection]' = Queue()
        self.tracking_queue: 'Queue[DataCollection]' = Queue()
        queue_for_segmentation: 'Queue[DataCollection]' = self.tracking_queue
        if self.use_pose:
            self.pose_queue: 'Queue[DataCollection]' = Queue()
            queue_for_segmentation = self.pose_queue
        self.segment_queue: 'Queue[DataCollection]' = Queue()
        self.segments: List[SegmentProducer] = [
            SegmentProducer(
                queue_for_segmentation,
                self.segment_queue,
                frame_pool,
                down_scale=down_scale,
                fast=fast,
                specific_bodypart=specific_bodypart
            )
            for _ in range(segment_processes)
        ]
        if self.use_pose:
            self.pose: PoseProducer = PoseProducer(
                self.tracking_queue, self.pose_queue, frame_pool)
        self.tracker: TrackProducer = TrackProducer(
            self.frame_queue,
            self.tracking_queue,
            frame_pool,
            down_scale=down_scale
        )
        self.cap = VideoCaptureProducer(
            self.frame_queue,
            camera_settings,
            frame_pool,
            skip_frames=skip_capture_frames
        )

    def start(self, handle_logs: bool = False) -> None:
        self.cap.start(handle_logs)
        self.tracker.start(handle_logs)
        if self.use_pose:
            self.pose.start(handle_logs)
        for segment in self.segments:
            segment.start(handle_logs)

    def get_frames(self) -> Generator[DataCollection, None, None]:
        while True:
            try:
                yield self.segment_queue.get(timeout=0.01)
            except queue.Empty:
                pass

    def stop(self) -> None:
        self.cap.stop()
        self.tracker.join()
        if self.use_pose:
            self.pose.join()
        for segment in self.segments:
            segment.stop()
        clear_queue(self.frame_queue, self.frame_pools)
        clear_queue(self.tracking_queue, self.frame_pools)
        if self.use_pose:
            clear_queue(self.pose_queue, self.frame_pools)
        clear_queue(self.segment_queue, self.frame_pools)


def clear_queue(
        clear_queue: Queue,
        frame_pools: Optional[Dict[Type, FramePool]] = None,
) -> None:
    try:
        if not frame_pools:
            while True:
                clear_queue.get_nowait()
        else:
            while True:
                data: DataCollection = clear_queue.get_nowait()
                for frame_type, frame_pool in frame_pools.items():
                    if frame_pool is not None and data.has(frame_type) \
                            and data.get(frame_type).using_shared_pool:
                        frame_data = data.get(frame_type)
                        assert isinstance(frame_data, FrameData)
                        frame_pool.free_frame(frame_data.frame)
    except queue.Empty:
        pass
    clear_queue.close()
