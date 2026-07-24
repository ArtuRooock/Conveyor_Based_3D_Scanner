import cv2 as cv
import numpy as np
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


def cutter(frame: cv.Mat,
           point1: tuple[int, int],
           point2: tuple[int, int],
           ) -> cv.Mat:
    x1, x2 = point1[0], point2[0]
    y = max(point1[1], point2[1])
    x1, x2 = sorted((x1, x2))
    x1 = max(0, x1)
    x2 = min(frame.shape[0], x2)
    return frame[0:y, x1:x2]


def compute_row_average_x_fast(gray: np.ndarray) -> np.ndarray:
    mask = gray > 0
    height, width = gray.shape
    x_indices = np.arange(width)

    sum_x = (mask * x_indices).sum(axis=1)
    count = mask.sum(axis=1)

    valid = count > 0
    avg_x = np.zeros(height, dtype=np.int32)
    avg_x[valid] = (sum_x[valid] / count[valid]).astype(np.int32)

    y_indices = np.arange(height)
    points = np.stack([avg_x[valid], y_indices[valid]], axis=1)

    return points


def laser_detector(gr: cv.Mat, lowerQ: int, upperQ: int, label="CAM"):
    gr = cv.bitwise_and(
        gr, gr, mask=cv.inRange(gr, lowerQ, upperQ))
    points = compute_row_average_x_fast(gr)
    cv.imshow(f"{label} - INPUT", gr)
    lasered = cv.Mat(np.ndarray(gr.shape, dtype=np.uint8))
    for point in points:
        cv.circle(lasered, point, 1, (255, 0, 0), 1)
    cv.imshow(f"{label} - LASERD", lasered)
    return points


FILENAME: str = Path("CloudBuilder", "camera.mp4")

if __name__ == "__main__":
    cap = cv.VideoCapture(FILENAME)
    if not cap.isOpened():
        raise RuntimeError("File not found")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 800)
    cap.set(cv.CAP_PROP_FPS, 120)

    ret, frame = cap.read()
    upper_point = get_click_coordinates(frame, "Upper cut")
    lower_point = get_click_coordinates(frame, "Lower cut")
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cutter(frame, lower_point, upper_point)

    while ret:
        points = laser_detector(frame, 250, 255)
        ret, frame = cap.read()
        if not ret:
            break
        cv.imshow("row", frame)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cutter(frame, lower_point, upper_point)
        cv.waitKey(1)
