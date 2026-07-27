import cv2 as cv
import numpy as np
from threading import Thread, Lock, Event


class Camera:
    def __init__(self, cam: str | int, max_queue_size: int = 240) -> None:
        self.camera = cam
        self.cap: cv.VideoCapture | None = None
        self.video_frame: np.ndarray | None = None
        self.lock = Lock()
        self.__stop_event__ = Event()
        self.__video_thread__: Thread | None = None
        self.frame_queue: list[cv.Mat] = []
        self.max_queue_size = max_queue_size

    def start_video(self):
        if self.cap is None:
            self.cap = cv.VideoCapture(self.camera)
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 800)
            self.cap.set(cv.CAP_PROP_FPS, 60)
        if not self.cap.isOpened():
            raise RuntimeError("No video")
        self.__video_thread__ = Thread(
            target=self.__capture_loop__, daemon=True)
        self.__video_thread__.start()

    def __capture_loop__(self):
        while not self.__stop_event__.is_set():
            ret, frame = self.cap.read()
            if not ret:
                # print("Lost video")
                continue
            with self.lock:
                self.video_frame = frame
                if len(self.frame_queue) > self.max_queue_size:
                    self.frame_queue.pop(0)
                self.frame_queue.append(frame)

    def snapshot(self) -> cv.Mat | None:
        cap = cv.VideoCapture(self.camera)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 800)
        cap.set(cv.CAP_PROP_FPS, 60)
        self.cap = cap
        frame = None
        while frame is None:
            ret, frame = cap.read()
            # if not ret:
            #     return None
        return frame

    def get_frame(self):
        with self.lock:
            return None if self.video_frame is None else self.video_frame.copy()

    def get_from_queue(self) -> cv.Mat | None:
        if len(self.frame_queue) == 0:
            return None
        return self.frame_queue.pop(0)

    def empty(self) -> bool:
        return len(self.frame_queue) == 0

    def stop(self):
        self.__stop_event__.set()
        if self.__video_thread__:
            self.__video_thread__.join()
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    cam = Camera(3)
    mat = cam.snapshot()
    cv.imshow("Win", mat)
    cv.waitKey(-1)