import os
import cv2
import numpy as np
import math


import cheker_params_little as cheker_params


t = 0
img_num=2
if t == 1:
    cam = "basler"
else:
    cam = "china"

if cam == "basler":
    img_shape = (1920, 1200)
else:
    img_shape = (1280, 800)

image = f"img\laser miniData\china_030.png"

DET_RADIUS = 120
Xp = 0
Yp = 0
Xp1 = 0
Yp1 = 0
PS = 0

def draw(img, corners, imgpts):
    """Draw Axes"""
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def click_and_crop(event, x, y, flags, param):
    global Xp, Yp, Xp1, Yp1
    if event == cv2.EVENT_LBUTTONUP:
        if PS == 0:
            Xp = x
            Yp = y
            print(PS, Xp, Yp)
        else:
            Xp1 = x
            Yp1 = y
            print(PS, Xp1, Yp1)


if __name__ == '__main__':
    calib = np.load(f"calib_{cam}.npz")

    print(calib['mtx'])
    mtx = calib['mtx']
    print(mtx)
    cy = mtx[1][2]

    cv2.namedWindow("select")
    cv2.setMouseCallback("select", click_and_crop)

    psns = []
    cam_ptss = []


    img = cv2.imread(image, cv2.IMREAD_COLOR)
    dst_ = cv2.undistort(img, calib['mtx'], calib['dist'], None, None)
    dst = cv2.cvtColor(dst_, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img', dst)
    k = cv2.waitKey(0)
    if k == ord('q'):
        print("Exit with 'q'")
        exit(2)

    ret, corn = cheker_params.get_corners(dst)
    if not ret:
        print("No corners found")
        exit(1)
    cv2.drawChessboardCorners(dst, cheker_params.CHECKERBOARD, corn, ret)
    cv2.imshow('img', dst)
    k = cv2.waitKey(0)
    if k == ord('q'):
        print("Exit with 'q'")
        exit(2)

    ret, rvec, tvec = cv2.solvePnP(cheker_params.corner_coordinates, corn, mtx, None)
    rot = cv2.Rodrigues(rvec)[0]

    pts = cv2.projectPoints(cheker_params.corner_coordinates, rvec, tvec, mtx, None)[0]


    rpoints = np.array([
        cheker_params.corner_coordinates[0][:2],
        cheker_params.corner_coordinates[cheker_params.CHECKERBOARD[0] - 1][:2],
        cheker_params.corner_coordinates[cheker_params.CHECKERBOARD[0] * cheker_params.CHECKERBOARD[1] - 1][:2],
        cheker_params.corner_coordinates[cheker_params.CHECKERBOARD[0] * (cheker_params.CHECKERBOARD[1] - 1)][:2]
    ])*DET_RADIUS
    imgpoints = np.array([
        corn[0],
        corn[cheker_params.CHECKERBOARD[0] - 1],
        corn[cheker_params.CHECKERBOARD[0] * cheker_params.CHECKERBOARD[1] - 1],
        corn[cheker_params.CHECKERBOARD[0] * (cheker_params.CHECKERBOARD[1] - 1)]
    ])

    hm = cv2.findHomography(imgpoints, rpoints, 0)

    homo_mat = hm[0]
    img_size = (int(rpoints[2][0]), int(rpoints[2][1]))
    pix_to_real = (((cheker_params.CHECKERBOARD[0] - 1) * cheker_params.SCALE_FACTOR) / rpoints[2][0],
                   ((cheker_params.CHECKERBOARD[1] - 1) * cheker_params.SCALE_FACTOR) / rpoints[2][1])
    checker_size = cheker_params.SCALE_FACTOR

    dstf = dst_.copy()

    dst_[:int(cy), : ] = (0,0,0)
    dst_[int(cy)+1: , :] = (0,0,0)

    dstf = cv2.warpPerspective(dstf, homo_mat, img_size)
    rm = cv2.warpPerspective(dst_, homo_mat, img_size)

    dstf = cv2.add(dstf, rm)
    while True:
        cv2.imshow('select', dstf)
        t = cv2.waitKey(0)
        if t == ord('1'):
            PS = 0
        elif t == ord('2'):
            PS = 1
        elif t == ord('n'):
            break


    def calx(y):
        return Xp + (y - Yp) * (Xp1 - Xp) / (Yp1 - Yp)


    for j in range(img_size[1]):
        if j%6 != 0:
            continue
        pos_checker_cm = np.array([
            calx(j) * pix_to_real[0],
            j * pix_to_real[1],
            0
        ])
        pos_camera_cm = np.dot(rot, pos_checker_cm) + tvec.T
        cam_pts = cv2.projectPoints(pos_checker_cm, rvec, tvec, mtx, None)[0][0][0]
        print('{'+f"{pos_camera_cm[0][0]}, {pos_camera_cm[0][1]}, {pos_camera_cm[0][2]}, {cam_pts[0]}, {cam_pts[1]}" +'},' )

        psns.append(
            pos_camera_cm[0]
        )
        cam_ptss.append(cam_pts)


    np.savez(f'angle_{cam}.npz', psns=psns, cam_ptss=cam_ptss)
    cv2.destroyAllWindows()

    print("Next: call CXX application 'optimize_laser_angle' to find laser angle params")


angle = math.degrees(math.atan2(Xp1 - Xp, Yp1 - Yp))
print(angle)
