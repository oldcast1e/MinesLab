
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 23:50:30 2023

@author: akhil_kk
"""

# Import required modules
import cv2
import numpy as np
import os
import glob
  
    
def stereoCalibrateCamera(camera_c1, camera_c2,camera_name,chessboard_box_size=1,chessboard_grid_size=(9,6),number_of_frames=50):
    
    # Define the dimensions of checkerboard
    CHECKERBOARD = chessboard_grid_size
    
    #chessboard square size in mm
    square_size=chessboard_box_size  
      
    # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS + 
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
      
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC     
 
    # Vector for 3D points
    threedpoints = []
      
    # Vector for 2D points
    twodpoints_c1 = []
    twodpoints_c2 = []
   
      
    #  3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0] 
                          * CHECKERBOARD[1], 
                          3), np.float32)
    
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                                   0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objectp3d *=square_size
    

    img_list_c1 =[] 
    img_list_c2 =[] 
    print("Align camera properly.. and press 'c' to capture")
    img_count=0
    while img_count<number_of_frames:
        dat1_rcved, img1 = camera_c1.read()  
        dat2_rcved, img2 = camera_c2.read()      
        cv2.imshow('img_c1',img1)
        cv2.imshow('img_c2',img2)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('c'):
            img_list_c1.append(img1)
            img_list_c2.append(img2)
            img_count +=1
            print(str(img_count)+" image captured")
        elif k == ord('x'):
            cv2.destroyAllWindows()
            print('capture terminated, ABORTING')
            return
    cv2.destroyAllWindows()
    
    for image1, image2 in zip(img_list_c1, img_list_c2):
        grayColor1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        grayColor2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret1, corners1 = cv2.findChessboardCorners(
                        grayColor1, CHECKERBOARD, 
                        cv2.CALIB_CB_ADAPTIVE_THRESH 
                        + cv2.CALIB_CB_FAST_CHECK + 
                        cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret2, corners2 = cv2.findChessboardCorners(
                        grayColor2, CHECKERBOARD, 
                        cv2.CALIB_CB_ADAPTIVE_THRESH 
                        + cv2.CALIB_CB_FAST_CHECK + 
                        cv2.CALIB_CB_NORMALIZE_IMAGE)
      
        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board
        if ret1 == True and ret2 == True :
            threedpoints.append(objectp3d)
      
            # Refining pixel coordinates
            # for given 2d points.
            cornersf_1 = cv2.cornerSubPix(
                grayColor1, corners1, (11, 11), (-1, -1), criteria)
            cornersf_2 = cv2.cornerSubPix(
                grayColor2, corners2, (11, 11), (-1, -1), criteria)
      
            twodpoints_c1.append(cornersf_1)
            twodpoints_c2.append(cornersf_2)
      
            # Draw and display the corners
            image1 = cv2.drawChessboardCorners(image1, 
                                              CHECKERBOARD, 
                                              cornersf_1, ret1)

            image2 = cv2.drawChessboardCorners(image2, 
                                              CHECKERBOARD, 
                                              cornersf_2, ret2)
      
            cv2.imshow('c1_corners',image1)
            cv2.imshow('c2_corners',image2)
            cv2.waitKey(1000)
      
    cv2.destroyAllWindows()
    
    #extract image shape 
    width= image1.shape[1]
    height= image1.shape[0]

    # Perform camera calibration by
    # passing the value of above found out 3D points (threedpoints)
    # and its corresponding pixel coordinates of the
    # detected corners (twodpoints)
    ret_1, k1, d1, r_1, t_1 = cv2.calibrateCamera(
        threedpoints, twodpoints_c1,(width, height), None, None)

    ret_2, k2, d2, r_2, t_2 = cv2.calibrateCamera(
        threedpoints, twodpoints_c2,(width, height), None, None)

    np.savez(camera_name+"c1.npz", k=k1,d=d1,r=r_1,t=t_1)
    np.savez(camera_name+"c2.npz", k=k2,d=d2,r=r_2,t=t_2)
    
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(threedpoints, twodpoints_c1, twodpoints_c2, k1, d1,
                                                                 k2, d2, (width, height), criteria = criteria, flags =   stereocalibration_flags)
    
    np.savez(camera_name+".npz", k1=k1,d1=d1,k2=k2,d2=d2,SR=R,ST=T)    
        


def getStereoCameraParameters(file_name):
    loaded_data = np.load(file_name)
    return loaded_data['k1'], loaded_data['d1'], loaded_data['k2'],loaded_data['d2'], loaded_data['SR'],loaded_data['ST']

def getgetStereoSingleCameraParameters(file_name):
    loaded_data = np.load(file_name)
    return loaded_data['k'], loaded_data['d'], loaded_data['r'],loaded_data['t']


#customCalibrateCamera(0,'lenovo_web_cam',24)
#print(getCameraParameters('lenovo_web_cam.npz'))
