
import cv2 as cv
import numpy as np
from dataclasses import dataclass
from pathlib import Path


def get_click_coordinates(img: cv.Mat, label: str):
    coords = {"x": None, "y": None}

    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            coords["x"] = x
            coords["y"] = y

    cv.imshow(label, img)
    cv.setMouseCallback(label, mouse_callback)

    while coords["x"] is None:
        if cv.waitKey(20) & 0xFF == 27:
            break

    cv.destroyAllWindows()

    if coords["x"] is None:
        return None

    return coords["x"], coords["y"]


def get_rot_mat_y(eta: float) -> np.ndarray:
    c, s = np.cos(eta), np.sin(eta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def get_rot_mat_z(eta: float) -> np.ndarray:
    c, s = np.cos(eta), np.sin(eta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def plane_norm(eta: float, theta: float) -> np.ndarray:
    pb = np.array([0, 0, 1])
    pb2 = np.array([0, -1, 0])
    My = get_rot_mat_y(theta)
    print(theta)
    Mz = get_rot_mat_z(eta)
    print(eta)
    d = np.cross(My.dot(Mz.dot(pb)), My.dot(Mz.dot(pb2)))
    return d


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


def load_calibration(conf_path: str, cam_name: str) -> CalibData:
    calib = np.load(f"CalibrationFiles/Cam1/calib_china1.npz")
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

    opt = np.load(f"CalibrationFiles/Cam1/result_step3_china1.npz")
    baf = opt["baf"]
    b = baf[0]
    th = baf[1] * (np.pi / 180.0)
    fi = baf[2] * (np.pi / 180.0)

    plane = np.load(f"CalibrationFiles/Cam1/plane_china1.npz")
    r_raw = plane["e"]
    t_raw = plane["pc"]
    R = r_raw  # same convention as the C++ R(i%3, i/3) fill
    T = t_raw

    pN = plane_norm(fi, th)

    return CalibData(fx=fx, cx=cx, cy=cy, mx=camera_mtx, my=camera_dist, b=b, R=R, T=T, pN=pN)


def triangulate_rows(
    # sub-pixel laser-line x per valid row (cropped-image coords)
    psx: np.ndarray,
    rows_y: np.ndarray,     # corresponding row indices (cropped-image coords)
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
    ], axis=1)  # (N, 3)

    # t = rays @ calib.pN                     # (N,)
    t = np.dot(calib.pN, rays.T)
    scale = (calib.b * calib.pN[0]) / t     # (N,)
    i_points = rays * scale[:, None]        # (N, 3)  -- getIntersectPoint
    print(f"i_points - {i_points}")
    # ------------------------

    move_vec = np.array([-pos, 0.0, 0.0])
    # n_points = (i_points - calib.T) @ calib.R.T + \
    #     move_vec  # R @ (iPoint - T) + moveVec

    n_points = np.dot(calib.R, (i_points - calib.T).T).T  # + move_vec
    print(f"t = {t}")
    print(f"scale = {scale}")
    print(f"pN = {calib.pN}")
    return i_points, n_points


DIR: str = "Cam2"
CAM = 0

cap = cv.VideoCapture(6)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv.CAP_PROP_FPS, 120)
ret, frame = cap.read()
calib = load_calibration(Path(DIR), "china2")
frame = cv.undistort(frame, calib.mx, calib.my, None, None)
point_A = get_click_coordinates(frame, "A")
point_B = get_click_coordinates(frame, "B")
print(f"{point_A}\n{point_B}")
i_points, n_points = triangulate_rows(
    np.asarray([point_A[0], point_B[0]]),
    np.asarray([point_A[1], point_B[1]]),
    0,
    0,
    calib,
    0
)
print(f"n_points - {n_points}")
print(np.linalg.norm(i_points[0] - i_points[1]))
print(np.linalg.norm(n_points[0] - n_points[1]))
