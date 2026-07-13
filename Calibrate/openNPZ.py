import numpy as np


data = np.load("calib_china.npz")
print(data.files)

rvecs = data["rvecs"]
tvecs = data["tvecs"]
mtx = data["mtx"]
dist = data["dist"]
img_shape = data["img_shape"]

print(rvecs)
# print(tvecs)
# print(mtx)
# print(dist)
# print(img_shape)
# baf = data['baf']
# # cam_ptss = data['cam_ptss']
# #
# print(baf)
# print(cam_ptss)
