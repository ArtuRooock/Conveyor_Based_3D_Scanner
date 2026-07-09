import numpy as np

import cv2


CHECKERBOARD = (5, 3)
SCALE_FACTOR = 1.5
corner_coordinates = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
corner_coordinates[:, :2] = np.indices(CHECKERBOARD).T.reshape(-1, 2)
corner_coordinates *= SCALE_FACTOR


def get_corners(g_image):
    ret, corners = cv2.findChessboardCornersSB(g_image, CHECKERBOARD)
    if not ret:
        return ret, corners

    return ret, cv2.cornerSubPix(g_image, corners, (21, 21), (-1, -1),
                                 (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                                  30, 0.001)).reshape(-1, 2)
