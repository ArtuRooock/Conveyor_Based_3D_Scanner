from ComReader import ComReader
from Camera import Camera
import cv2 as cv
import numpy as np
from typing import Any
from time import time

from dataclasses import dataclass


@dataclass
class FrameAndOffset:
    frame: cv.Mat
    speed: dict[str, Any]
    timestamp: float


class MotionData(Camera):
    def __init__(self, cam: str | int, reader: ComReader, max_queue_size: int = 240) -> None:
        super().__init__(cam, max_queue_size)
        self.reader = reader
        self.frame_queue: list[FrameAndOffset] = []

    def __capture_loop__(self):
        while not self.__stop_event__.is_set():
            ret, frame = self.cap.read()
            if not ret:
                # print("Lost video")
                continue
            with self.lock:
                self.video_frame = frame
                data = self.reader.data
                if len(self.frame_queue) > self.max_queue_size:
                    self.frame_queue.pop(0)
                self.frame_queue.append(FrameAndOffset(frame, data, time()))

    def get_full_from_queue(self) -> FrameAndOffset | None:
        if len(self.frame_queue) == 0:
            return None
        return self.frame_queue.pop(0)

    def get_from_queue(self) -> cv.Mat | None:
        return self.get_full_from_queue().frame
