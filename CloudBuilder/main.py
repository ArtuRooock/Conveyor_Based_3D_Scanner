from MotionData import MotionData, FrameAndOffset
from CloudBuilder import CloudBuilder, CalibData
from LaserDetector import laser_detector, cutter, get_click_coordinates
import cv2 as cv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from ComReader import ComReader
import keyboard
from time import time
import numpy as np
import json


def save_to_pts(points_xyz: np.ndarray, filename: str):
    if points_xyz.shape[1] != 3:
        raise ValueError("Expected array shape (N, 3)")
    np.savetxt(filename, points_xyz, fmt='%.6f', delimiter=' ',
               header='', comments='', encoding='utf-8')


class MockReader(ComReader):
    @property
    def data(self):
        return {OFFSET_KEY: float("0.02")}


@dataclass
class CamSettings:
    calib_path: Path
    cam_name: str
    cut_points: NDArray  # (2, 2)
    laser_low: int
    laser_up: int


CAM_1 = 6
CAM_2 = 8
CALIB_PATH = Path("Calibrate", "CalibrationFiles")
OUTPUT_PATH = Path("Out")
ALL_POINTS: list[NDArray] = []
COM = "/dev/ttyUSB0"
OFFSET_KEY = "speed_mms"
INTER_CAM_DISTANCE = -15.035101648521511

CLOUD_1, CLOUD_2 = [], []


def calibrate(settings: CamSettings) -> CalibData:
    calib_data = CloudBuilder.load_calibration(
        settings.calib_path, settings.cam_name)
    return calib_data


def proccess_img(cam: MotionData,
                 settings: CamSettings,
                 calib: CalibData,
                 pos: float,
                 current_timestamp: float,
                 label="CAM") -> tuple[NDArray, float]:
    data = None
    while data is None or data.frame is None or data.speed is None:
        data: FrameAndOffset = cam.get_full_from_queue()
    frame = data.frame
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.remap(frame, calib.mx, calib.my, cv.INTER_LINEAR)
    frame = cutter(frame, settings.cut_points[0], settings.cut_points[1])
    if pos is None:
        cv.imshow("frame", frame)
        cv.waitKey(5)
        pass
    points = laser_detector(frame, settings.laser_low,
                            settings.laser_up, label)
    time_delta = data.timestamp - current_timestamp
    pos_offset: float = data.speed[OFFSET_KEY] * time_delta
    points3D = CloudBuilder.triangulate_rows(points[:, 0],
                                             points[:, 1],
                                             min(settings.cut_points[:, 0]), 0,
                                             calib, pos + pos_offset)
    return points3D, pos + pos_offset, data.timestamp


def get_rect(frame: cv.Mat, label: str) -> NDArray:
    point_A = get_click_coordinates(frame, f"{label} A")
    point_B = get_click_coordinates(frame, f"{label} B")
    return np.asarray([point_A, point_B])


if __name__ == "__main__":
    reader: ComReader = ComReader(COM)
    reader.start_read()

    cam1 = MotionData(CAM_1, reader)
    cam2 = MotionData(CAM_2, reader)

    cold_frame1 = cam1.snapshot()
    rect1 = get_rect(cold_frame1, "cam1")
    cam1_settings = CamSettings(
        Path(CALIB_PATH, "Cam1"), "china1", rect1, 150, 255)

    cold_frame2 = cam2.snapshot()
    rect2 = get_rect(cold_frame2, "cam2")
    cam2_settings = CamSettings(
        Path(CALIB_PATH, "Cam2"), "china2", rect2, 150, 255)

    cam1_calib = calibrate(cam1_settings)
    cam2_calib = calibrate(cam2_settings)

    cam1.start_video()
    cam2.start_video()

    cam1_pos = 0.0
    cam2_pos = cam1_pos + INTER_CAM_DISTANCE

    cam1_current_timestamp = time()
    cam2_current_timestamp = time()

    while cam1.empty() and cam2.empty():
        pass

    while not keyboard.is_pressed('q'):
        cam1_points, cam1_pos, cam1_current_timestamp = proccess_img(
            cam1, cam1_settings, cam1_calib, cam1_pos,
            cam1_current_timestamp, "CAM1")

        cam2_points, cam2_pos, cam2_current_timestamp = proccess_img(
            cam2, cam2_settings, cam2_calib, cam2_pos,
            cam2_current_timestamp, "CAM2")

        if cam1_points is not None:
            CLOUD_1.append(cam1_points)
            ALL_POINTS.append(cam1_points)

        if cam2_points is not None:
            CLOUD_2.append(cam2_points)
            ALL_POINTS.append(cam2_points)

        print(json.dumps(ALL_POINTS[-1].tolist()))

    save_to_pts(np.vstack(ALL_POINTS), "tvar.pts")
    save_to_pts(np.vstack(CLOUD_1), "cam1.pts")
    save_to_pts(np.vstack(CLOUD_2), "cam2.pts")

    reader.stop_read()
