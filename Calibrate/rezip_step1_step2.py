import zipfile

def to_legacy_npz(src_path, dst_path):
    with zipfile.ZipFile(src_path, 'r') as zin:
        with zipfile.ZipFile(dst_path, 'w',
                              compression=zipfile.ZIP_STORED,
                              allowZip64=False) as zout:
            for item in zin.infolist():
                zout.writestr(item.filename, zin.read(item.filename))

# to_legacy_npz('CalibrationFiles/Cam1/calib_china1.npz', 'CalibrationFiles/Cam1/calib_china1_f.npz')
# to_legacy_npz('CalibrationFiles/Cam1/pts_china1.npz', 'CalibrationFiles/Cam1/pts_china1_f.npz')
# to_legacy_npz('CalibrationFiles/Cam2/calib_china2.npz', 'CalibrationFiles/Cam2/calib_china2_f.npz')
# to_legacy_npz('CalibrationFiles/Cam2/pts_china2.npz', 'CalibrationFiles/Cam2/pts_china2_f.npz')
# to_legacy_npz('angle_china.npz', 'angle_china_fixed.npz')

to_legacy_npz('CalibrationFiles/Cam1/angle_china1.npz', 'CalibrationFiles/Cam1/angle_china1_f.npz')
to_legacy_npz('CalibrationFiles/Cam2/angle_china2.npz', 'CalibrationFiles/Cam2/angle_china2_f.npz')
#
# to_legacy_npz('CalibrationFiles/Cam2_Fix/calib_china2.npz', 'CalibrationFiles/Cam2_Fix/calib_china2_f.npz')
# to_legacy_npz('CalibrationFiles/Cam2_Fix/pts_china2.npz', 'CalibrationFiles/Cam2_Fix/pts_china2_f.npz')
# to_legacy_npz('CalibrationFiles/Cam2/angle_china2.npz', 'CalibrationFiles/Cam2/angle_china2_f.npz')