import queue
from multiprocessing import Queue
from multiprocessing.sharedctypes import Synchronized
from typing import Generator, List, Optional

from frame.camera import CaptureSettings
from frame.producer import VideoCaptureProducer
from frame.shared import FramePool
from pipeline.data import DataCollection
from pose.producer import PoseProducer
from segmentation.producer import SegmentProducer
from tracking.producer import TrackProducer


class FrameProcessingPipeline:
    def __init__(
        self,
        segment_processes: int = 2,
        down_scale: Optional[float] = None,
        fast: bool = True,
        camera_settings: Optional[CaptureSettings] = None,
        frame_pool: Optional[FramePool] = None,
        specific_bodypart: Optional[Synchronized] = None,
        use_pose: bool = False
    ) -> None:
        self.use_pose = use_pose

        self.frame_queue: Queue[DataCollection] = Queue()
        self.tracking_queue: Queue[DataCollection] = Queue()
        queue_for_segmentation: Queue[DataCollection] = self.tracking_queue
        if self.use_pose:
            self.pose_queue: Queue[DataCollection] = Queue()
            queue_for_segmentation = self.pose_queue
        self.segment_queue: Queue[DataCollection] = Queue()
        self.segments: List[SegmentProducer] = [
            SegmentProducer(
                queue_for_segmentation,
                self.segment_queue,
                down_scale,
                fast,
                frame_pool,
                specific_bodypart
            )
            for _ in range(segment_processes)
        ]
        if self.use_pose:
            self.pose: PoseProducer = PoseProducer(
                self.tracking_queue, self.pose_queue, frame_pool=frame_pool)
        self.tracker: TrackProducer = TrackProducer(
            self.frame_queue, self.tracking_queue, down_scale, frame_pool)
        self.cap = VideoCaptureProducer(
            self.frame_queue, camera_settings, frame_pool)

    def start(self) -> None:
        self.cap.start()
        self.tracker.start()
        if self.use_pose:
            self.pose.start()
        for segment in self.segments:
            segment.start()

    def get_frames(self) -> Generator[DataCollection, None, None]:
        while True:
            try:
                yield self.segment_queue.get(timeout=0.01)
            except queue.Empty:
                pass

    def stop(self) -> None:
        self.cap.stop()
        self.tracker.stop()
        if self.use_pose:
            self.pose.stop()
        for segment in self.segments:
            segment.stop()
