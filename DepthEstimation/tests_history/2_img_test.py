#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 21:44:50 2023

@author: akhil_kk
"""
import cv2
import numpy as np
import math
from CustomCalibrateCamera import Calib_Camera


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])



# params for corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
  
# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              10, 0.03))


img1=cv2.imread("imageY_1.jpg")
img2=cv2.imread("imageY_2.jpg")

img1_g = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_g = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# Create a mask image for drawing purposes
mask = np.zeros_like(img1)


p0 = cv2.goodFeaturesToTrack(img1_g, mask = None,**feature_params)
# calculate optical flow
p1, st, err = cv2.calcOpticalFlowPyrLK(img1_g,img2_g,p0, None,**lk_params)


# Select good points
good_new = p1[st == 1]
good_old = p0[st == 1]
print(good_new)

st_img=np.hstack((img1,img2))
#cv2.imshow("stacked",st_img)
#cv2.waitKey(1200)
img_shape=img1.shape

# draw the tracks
for i, (new, old) in enumerate(zip(good_new, 
                                   good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    
    
    img = cv2.line(st_img, (int(a+img_shape[1]), b), (c, d),
                    (0,0,255), 2)
    print(i)
    #img = cv2.circle(st_img, (int(a+img_shape[1]), b), 5, (255-(i*10),0,0), -1)
    #img = cv2.circle(img, (c, d), 5, (255-(i*10),0,0), -1)
      
    #frame = cv2.circle(img2, (a, b), 5,
    #                   (255,0,0), -1)
    
    #img = cv2.add(img2, mask)
  
    cv2.imshow('frame', img)
      
    k = cv2.waitKey(30)
    if k == 27:
        break



cameraMatrix,_,_,_ = Calib_Camera.getCameraParameters('CustomCalibrateCamera/lenovo_web_cam.npz')

ret, out_arr=cv2.findEssentialMat(np.asarray(good_new), np.asarray(good_old), cameraMatrix)
print(ret)
print(out_arr)
r1,r2,t=cv2.decomposeEssentialMat(ret)
print(r1)
print(r2)
print(t)

print("euler angles")
print(rotationMatrixToEulerAngles(r1))

print("euler angles")
print(rotationMatrixToEulerAngles(r2))

