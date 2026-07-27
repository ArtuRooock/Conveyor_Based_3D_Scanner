import os
import cv2
import numpy as np

import cheker_params_little as cheker_params


t = 1
if t == 0:
    cam = "basler"
else:
    cam = "china"

if cam == "basler":
    img_shape = (1920, 1200)
elif cam == "china":
    img_shape = (1280, 800)


def draw(img, corners, imgpts):
    """Draw Axes"""
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


DET_RADIUS = 150
Xp = 0
Yp = 0


def click_and_crop(event, x, y, flags, param):
    global Xp, Yp
    if event == cv2.EVENT_LBUTTONUP:
        Xp = x
        Yp = y
        print(Xp, Yp)



if __name__ == '__main__':
    numCam = int(input("Number of camera: "))
    numStep = int(input("Number of step: "))
    directory = f"Frames/Cam{numCam}_Step{numStep}"
    calib = np.load(f"CalibrationFiles/Cam{numCam}/calib_{cam}{numCam}.npz")
    print(calib['mtx'])
    mtx = calib['mtx']

    print(mtx)
    cy = mtx[1][2]
    print(cy)

    cv2.namedWindow("select")
    cv2.setMouseCallback("select", click_and_crop)

    imgps = os.listdir(directory)
    print(imgps)
    psns = []
    cam_ptss = []

    for imgp in imgps:
        if not imgp.startswith(cam):
            continue

        img = cv2.imread(os.path.join(directory, imgp), cv2.IMREAD_COLOR)
        dst_ = cv2.undistort(img, calib['mtx'], calib['dist'], None, None)
        dst = cv2.cvtColor(dst_, cv2.COLOR_BGR2GRAY)
        ret, corn = cheker_params.get_corners(dst)
        if not ret:
            print("No corners found")
            continue
        cv2.drawChessboardCorners(dst, cheker_params.CHECKERBOARD, corn, ret)
        cv2.imshow('img', dst)
        cv2.imwrite('relt.jpeg', dst)
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

        #dst_[:] = (0,0,0)
        #dst_[int(cy):int(cy)+1, : ] = (255,255,255)
        dstf[int(cy)-1:int(cy)+2, : ] = (0,0,0)
        dstf = cv2.warpPerspective(dstf, homo_mat, img_size)
        #rm = cv2.warpPerspective(dst_, homo_mat, img_size)

        skip = False
        #dstf = cv2.add(dstf, rm)

        while True:
            cv2.imshow('select', dstf)
            ch = cv2.waitKey(0)
            if ch == ord('n'):
                break
            if ch == ord('z'):
                skip = True
                break

        if skip:
            print("Skipppppped")
            continue

        pos_checker_cm = np.array([
            Xp * pix_to_real[0],
            Yp * pix_to_real[1],
            0
        ])

        pos_camera_cm = np.dot(rot, pos_checker_cm) + tvec.T
        cam_pts = cv2.projectPoints(pos_checker_cm, rvec, tvec, mtx, None)[0][0][0]

        print("POS_CM: ", pos_camera_cm[0], "CAM_PTS:", cam_pts)
        psns.append(
            pos_camera_cm[0]
        )
        cam_ptss.append(cam_pts)

    for i in range(len(psns)):
        print(
        '{' + f"{psns[i][0]}, {psns[i][1]}, {psns[i][2]}, {cam_ptss[i][0]}, {cam_ptss[i][1]}" + '},')

    np.savez(f'CalibrationFiles/Cam{numCam}/pts_{cam}{numCam}.npz', psns=psns, cam_ptss=cam_ptss)
    cv2.destroyAllWindows()

    print("Next: call CXX application 'optimize_camera_angle_baseline' to find baseline and laser angle params")

