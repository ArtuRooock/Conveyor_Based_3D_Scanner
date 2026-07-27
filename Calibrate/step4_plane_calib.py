import os
import cv2
import numpy as np
import scipy
from skspatial.objects import Plane, Point

import cheker_params_little as cheker_params

t = 1
if t == 0:
    cam = "basler"
else:
    cam = "china"

if cam == "basler":
    img_shape = (1920, 1200)
else:
    img_shape = (1280, 800)


def get_My(thetta):
    return np.array([
        [np.cos(thetta), 0, np.sin(thetta)],
        [0, 1, 0],
        [-np.sin(thetta), 0, np.cos(thetta)]
    ])


def get_Mz(alpha):
    return np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])


def intersect_point(ray, thetta, alpha, b):
    Mz = get_Mz(alpha)
    My = get_My(thetta)

    pb = np.array([0, 0, 1])
    pb2 = np.array([0, -1, 0])
    d = np.cross(My.dot(Mz.dot(pb)), My.dot(Mz.dot(pb2)))
    t = d.T.dot(ray)
    if t == 0:
        return [-100, -100, -100]

    return (b*d[0]/t)*ray

def get_camera_rotated_ray_vec(x_, y_, f_):
    return np.array([
        x_, y_, f_
    ])


def calc_z_by_point(x_, y_, f_, point_):
    alpha = np.arctan(np.sqrt(x_**2+y_**2)/f_)
    return np.linalg.norm(point_)*np.cos(alpha)


def calc_ptr(x_, y_, cx_, cy_, fx_, b_, th_, etta_):
    xp = x_ - cx_
    yp = y_ - cy_
    ray = get_camera_rotated_ray_vec(xp, yp, fx_)
    pt = intersect_point(ray, th_, etta_, b_)
    zr = calc_z_by_point(xp, yp, fx_, pt)
    xr = xp*zr / fx_
    yr = yp*zr / fx_
    return np.array([xr, yr, zr])


PS = 0

Pstart = None
Pend = None

Pinter1 = (0, 0)
Pinter2 = (0, 0)

def draw(img, corners, imgpts):
    """Draw Axes"""
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def click_and_crop(event, x, y, flags, param):
    global Pstart, Pend, Pinter1, Pinter2
    if event == cv2.EVENT_LBUTTONUP:
        if PS == 0:
            Pstart = (x, y)
            print(PS, Pstart)
        elif PS == 1:
            Pend = (x, y)
            print(PS, Pend)
        elif PS == 2:
            Pinter1 = (x, y)
            print(PS, Pinter1)
        elif PS == 3:
            Pinter2 = (x, y)
            print(PS, Pinter2)


def angle_between(v1, v2):
    dot_pr = v1.dot(v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)

    return np.rad2deg(np.arccos(dot_pr / norms))

if __name__ == '__main__':
    numCam = int(input("Number of camera: "))
    directory = "imgs"

    image_nolaser_move = f"Frames/Cam{numCam}_Step3/{cam}_008.png"
    image_nolaser = f"Frames/Cam{numCam}_Step3/{cam}_006.png"
    image_laser = f"Frames/Cam{numCam}_Step3/{cam}_002.png"

    calib = np.load(f'CalibrationFiles/Cam{numCam}/calib_{cam}{numCam}_f.npz')
    las = np.load(f"CalibrationFiles/Cam{numCam}/result_step3_{cam}{numCam}.npz")["baf"]

    print(las)

    print(calib['mtx'])
    mtx = calib['mtx']
    cx = mtx[0][2]
    cy = mtx[1][2]
    fx = mtx[0][0]
    b = las[0]
    th = np.radians(las[1])
    etta = np.radians(las[2])

    print("Laser params: ")
    print(f"cx={cx};cy={cy};fx={fx}")
    print(f"b={b};th={th};etta={etta}")


    cv2.namedWindow("select")
    cv2.setMouseCallback("select", click_and_crop)

    psns = []
    cam_ptss = []

    img2 = cv2.imread(image_nolaser_move, cv2.IMREAD_COLOR)
    dst2_ = cv2.undistort(img2, calib['mtx'], calib['dist'], None, None)
    dst2 = cv2.cvtColor(dst2_, cv2.COLOR_BGR2GRAY)
    ret, corn2 = cheker_params.get_corners(dst2)
    if not ret:
        print("No corners found 1 ")
        exit(1)
    ret2, rvec2, tvec2 = cv2.solvePnP(cheker_params.corner_coordinates, corn2, mtx, None)
    rot2 = cv2.Rodrigues(rvec2)[0]
    pos_camera_cm_move = np.dot(rot2, cheker_params.corner_coordinates.T).T + tvec2.T
    print(pos_camera_cm_move)

    img = cv2.imread(image_nolaser, cv2.IMREAD_COLOR)
    dst_ = cv2.undistort(img, calib['mtx'], calib['dist'], None, None)
    dst = cv2.cvtColor(dst_, cv2.COLOR_BGR2GRAY)
    ret, corn = cheker_params.get_corners(dst)
    if not ret:
        print("No corners found img1 2")
        exit(1)
    ret, rvec, tvec = cv2.solvePnP(cheker_params.corner_coordinates, corn, mtx, None)
    rot = cv2.Rodrigues(rvec)[0]
    pos_camera_cm = np.dot(rot, cheker_params.corner_coordinates.T).T + tvec.T
    print(pos_camera_cm)


    img_laser = cv2.imread(image_laser, cv2.IMREAD_COLOR)
    dst_laser_ = cv2.undistort(img_laser, calib['mtx'], calib['dist'], None, None)
    dst_laser = cv2.cvtColor(dst_laser_, cv2.COLOR_BGR2GRAY)

    while(1):
        cv2.imshow('select', dst_laser)
        k = cv2.waitKey(0)
        if k == ord('q'):
            print("Exit with 'q'")
            exit(2)
        elif k == ord('1'):
            PS = 0
        elif k == ord('2'):
            PS = 1
        elif k == ord('3'):
            PS = 2
        elif k == ord('4'):
            PS = 3
        elif k == ord('n'):
            break

    cv2.drawChessboardCorners(dst_, cheker_params.CHECKERBOARD, corn, ret)
    cv2.imshow('c1', dst_)
    cv2.waitKey(0)

    cv2.drawChessboardCorners(dst2_, cheker_params.CHECKERBOARD, corn2, ret2)
    cv2.imshow('c1', dst2_)
    cv2.waitKey(0)

    print(f"Interesting area: {Pinter1}: {Pinter2[0] - Pinter1[0]}, {Pinter2[1] - Pinter1[1]}")

    pts_start = calc_ptr(Pstart[0], Pstart[1], cx, cy, fx, b, th, etta)
    print(f"Calc PO = {pts_start}")

    pts_end = calc_ptr(Pend[0], Pend[1], cx, cy, fx, b, th, etta)
    print(f"Calc P1 = {pts_end}")

    conv_center = (pts_end + pts_start) / 2

    print(f"Conv_center = {conv_center}")

    mvec = np.array([0., 0., 0.])
    npoints = len(pos_camera_cm)

    for i in range(npoints):
        mvec += (pos_camera_cm_move[i] - pos_camera_cm[i]) / npoints

    mvec /= np.linalg.norm(mvec)
    print(f"Calc mvec = {mvec}")

    X = []
    Y = []
    Z = []
    for i in range(len(pos_camera_cm)):
        X.append(pos_camera_cm[i][0])
        Y.append(pos_camera_cm[i][1])
        Z.append(pos_camera_cm[i][2])

    for i in range(len(pos_camera_cm_move)):
        X.append(pos_camera_cm_move[i][0])
        Y.append(pos_camera_cm_move[i][1])
        Z.append(pos_camera_cm_move[i][2])

    A = np.c_[X, Y, np.ones(len(X))]
    C, _, _, _ = scipy.linalg.lstsq(A, Z)  # coefficients

    PCoef = np.array([C[0], C[1], -1, C[2]])
    nx = C[0]
    ny = C[1]
    nz = -1
    v1 = np.array([
        nx, ny, nz
    ])
    ey = v1 / np.linalg.norm(v1)

    print(f"ey: ({ey})")
    ang = angle_between(ey, np.array([0, 0, 1]))
    if ang < 90:
        print("Wrong direction ey!!!", ang)
        ey *= -1
        print("Repaired: ", angle_between(ey, np.array([0, 0, 1])), ey)

    plane = Plane(point=[0, 0, C[2]], normal=ey)
    point = Point(np.array([0, 0, C[2]]) + mvec)
    point_projected = plane.project_point(point)
    print("PP:", point_projected)
    pb = np.array(point_projected) - np.array([0, 0, C[2]])
    print("Diff proj/est:", pb, mvec)
    ex = pb / np.linalg.norm(pb)
    print(f"Estimated ex: {ex}")

    ez = np.cross(
        ex, ey
    )
    ez /= np.linalg.norm(ez)

    print(f"{angle_between(ex, ey)} , {angle_between(ex, ez)} , {angle_between(ey, ez)}")

    print(f"EZ: ({ez})")
    eN = np.array([
        [ex[0], ey[0], ez[0]],
        [ex[1], ey[1], ez[1]],
        [ex[2], ey[2], ez[2]]
    ])

    print(f"{cam} -> conv:")
    print(f"{eN}")
    print("T:")
    print(f"{conv_center}")

    np.savez(f'CalibrationFiles/Cam{numCam}/plane_{cam}{numCam}.npz', e=eN, pc=conv_center)
