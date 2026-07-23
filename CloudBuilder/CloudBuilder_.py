from Camera import Camera
import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class CalibData:
    fx: float
    cx: float
    cy: float
    mx: np.ndarray          # undistort remap X
    my: np.ndarray          # undistort remap Y
    b: float                # laser baseline
    R: np.ndarray           # plane rotation (3x3)
    T: np.ndarray           # plane translation (3,)
    pN: np.ndarray          # laser plane normal (3,)


def get_rot_mat_y(eta: float) -> np.ndarray:
    c, s = np.cos(eta), np.sin(eta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def get_rot_mat_z(eta: float) -> np.ndarray:
    c, s = np.cos(eta), np.sin(eta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def plane_norm(eta: float, theta: float) -> np.ndarray:
    m = np.array([[0.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    t = get_rot_mat_y(theta) @ get_rot_mat_z(eta) @ m
    return np.cross(t[:, 0], t[:, 1])


class CloudBuilder:

    @staticmethod
    def load_calibration(conf_path: str, cam_name: str) -> CalibData:
        calib = np.load(f"{conf_path}/calib_{cam_name}.npz")
        camera_mtx = calib["mtx"]
        camera_dist = calib["dist"]
        img_shape = calib["img_shape"]
        fx = camera_mtx[0, 0]
        cx = camera_mtx[0, 2]
        cy = camera_mtx[1, 2]

        mx, my = cv.initUndistortRectifyMap(
            camera_mtx, camera_dist, None, camera_mtx,
            (int(img_shape[0]), int(img_shape[1])), cv.CV_32FC1,
        )

        opt = np.load(f"{conf_path}/result_step3_{cam_name}.npz")
        baf = opt["baf"]
        b = baf[0]
        th = baf[1] * (np.pi / 180.0)
        fi = baf[2] * (np.pi / 180.0)

        plane = np.load(f"{conf_path}/plane_{cam_name}.npz")
        r_raw = plane["e"]
        t_raw = plane["pc"]
        R = r_raw
        T = t_raw

        pN = plane_norm(fi, th)

        return CalibData(fx=fx, cx=cx, cy=cy, mx=mx, my=my, b=b, R=R, T=T, pN=pN)

    @staticmethod
    def triangulate_rows(
        psx: np.ndarray,
        rows_y: np.ndarray,
        start_x: int,
        start_y: int,
        calib: CalibData,
        pos: float,
    ) -> np.ndarray:
        if psx.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        rays = np.stack([
            start_x + psx - calib.cx,
            start_y + rows_y - calib.cy,
            np.full_like(psx, calib.fx),
        ], axis=1)

        t = np.dot(calib.pN, rays.T)
        scale = (calib.b * calib.pN[0]) / t
        i_points = rays * scale[:, None]

        move_vec = np.array([-pos, 0.0, 0.0])

        n_points = np.dot(calib.R.T, (i_points - calib.T).T).T + \
            move_vec
        return n_points
