import cv2
import numpy as np
from matplotlib import pyplot as plt
from CustomCalibrateCamera.Stereo_Calib_Camera import stereoCalibrateCamera
from CustomCalibrateCamera.Stereo_Calib_Camera import getStereoCameraParameters
from CustomCalibrateCamera.Stereo_Calib_Camera import getgetStereoSingleCameraParameters

import Camera.jetsonCam as jetCam
cam1 = jetCam.jetsonCam()
cam2 = jetCam.jetsonCam()


cam1.open(sensor_id=1,
          sensor_mode=3,
          flip_method=0,
          display_height=540,
          display_width=960,
        )
cam2.open(sensor_id=0,
          sensor_mode=3,
          flip_method=0,
          display_height=540,
          display_width=960,
        )

cam1.start()
cam2.start()
stereoCalibrateCamera(cam1,cam2,'CustomCalibrateCamera/jetson_stereo_8MP',24)
lod_data = getStereoCameraParameters('CustomCalibrateCamera/jetson_stereo_8MP.npz')
lod_datac1 = getgetStereoSingleCameraParameters('CustomCalibrateCamera/jetson_stereo_8MPc1.npz')
lod_datac2 = getgetStereoSingleCameraParameters('CustomCalibrateCamera/jetson_stereo_8MPc2.npz')

camera_matrix_left = lod_data[0]
dist_coeffs_left =  lod_data[1]
camera_matrix_right =  lod_data[2]
dist_coeffs_right =  lod_data[3]
R =  lod_data[4]
T =  lod_data[5]

#print camera matrix
print(lod_datac1[0])
print(lod_datac2[0])
print(lod_datac1[1])
print(lod_datac2[1])


cv2.destroyAllWindows()
cam1.stop()
cam2.stop()
cam1.release()
cam2.release()


