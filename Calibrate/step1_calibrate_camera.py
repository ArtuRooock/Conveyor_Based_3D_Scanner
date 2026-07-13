import os
import cv2
import numpy as np

import cheker_params_little

t = 1
if t == 0:
    cam = "basler"
else:
    cam = "china"

if cam == "basler":
    img_shape = (1920, 1200)
else:
    img_shape = (1280, 800)


if __name__ == '__main__':
    print("Loading images with checkerboard")
    print(img_shape)
    directory = "img/calib"
    corners = []
    objpoints = []
    lf = None
    for filename in os.listdir(directory):
        if not filename.startswith(cam):
            continue
        f = os.path.join(directory, filename)
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        ret, corn = cheker_params_little.get_corners(img)
        if not ret:
            continue
        cv2.drawChessboardCorners(img, cheker_params_little.CHECKERBOARD, corn, ret)
        cv2.imshow('img', img)
        k = cv2.waitKey(0)
        if k == ord('k'):
            lf = filename
            k = ord('s')
        if k == ord('s'):
            corners.append(corn)
            objpoints.append(cheker_params_little.corner_coordinates)
            print("appended")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, corners, img_shape, None, None,
                                                       flags=cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT)
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_shape, 1, img_shape)
    print("ret L:", ret)
    print(mtx)
    # print(newcameramtx)
    print("")

    np.savez(f"calib_{cam}.npz",
             rvecs=rvecs, tvecs=tvecs, mtx=mtx, dist=dist, img_shape=img_shape)
    f = os.path.join(directory, lf)
    print(f)
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    dst_ = cv2.undistort(img, mtx, dist, None, None)
    cv2.imwrite('rslt.jpeg', dst_)

