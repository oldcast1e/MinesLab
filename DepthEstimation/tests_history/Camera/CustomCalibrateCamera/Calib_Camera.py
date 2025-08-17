
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
  
  

  

    
def customCalibrateCamera(camera_device,camera_name,chessboard_box_size=1,chessboard_grid_size=(9,6),number_of_frames=30):
    
    # Define the dimensions of checkerboard
    CHECKERBOARD = chessboard_grid_size
    
    #chessboard square size in mm
    square_size=chessboard_box_size  
      
    # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS + 
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
      
      
    # Vector for 3D points
    threedpoints = []
      
    # Vector for 2D points
    twodpoints = []
   
      
    #  3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0] 
                          * CHECKERBOARD[1], 
                          3), np.float32)
    
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                                   0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objectp3d *=square_size
    
    #start camera stream
    cap = cv2.VideoCapture(camera_device)

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream")
        cap.release()
    img_list =[] 
    print("Align camera properly.. and press 'c' to capture")
    img_count=0
    while img_count<number_of_frames:
        dat_rcved, img = cap.read()        
        cv2.imshow('img',img)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('c'):
            img_list.append(img)
            img_count +=1
            print(str(img_count)+" image captured")
        elif k == ord('x'):
            cap.release()
            cv2.destroyAllWindows()
            print('capture terminated, ABORTING')
            return
    cv2.destroyAllWindows()
    cap.release()
    
    for image in img_list:
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if dat_rcved is False:
            break
        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
                        grayColor, CHECKERBOARD, 
                        cv2.CALIB_CB_ADAPTIVE_THRESH 
                        + cv2.CALIB_CB_FAST_CHECK + 
                        cv2.CALIB_CB_NORMALIZE_IMAGE)
      
        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board
        if ret == True:
            threedpoints.append(objectp3d)
      
            # Refining pixel coordinates
            # for given 2d points.
            corners2 = cv2.cornerSubPix(
                grayColor, corners, (11, 11), (-1, -1), criteria)
      
            twodpoints.append(corners2)
      
            # Draw and display the corners
            image = cv2.drawChessboardCorners(image, 
                                              CHECKERBOARD, 
                                              corners2, ret)
      
            cv2.imshow('detected_corners',image)
            cv2.waitKey(10)
      
    cv2.destroyAllWindows()
    
    # Perform camera calibration by
    # passing the value of above found out 3D points (threedpoints)
    # and its corresponding pixel coordinates of the
    # detected corners (twodpoints)
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints,image.shape[:2], None, None)
    
    np.savez(camera_name+".npz", K_matrix=matrix, Dist=distortion, r_vecs=r_vecs, t_vecs=t_vecs)    
        


def getCameraParameters(file_name):
    loaded_data = np.load(file_name)
    return loaded_data['K_matrix'], loaded_data['Dist'], loaded_data['r_vecs'],loaded_data['t_vecs']



#customCalibrateCamera(0,'lenovo_web_cam',24)
#print(getCameraParameters('lenovo_web_cam.npz'))