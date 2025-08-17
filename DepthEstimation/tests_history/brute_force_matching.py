#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 11:24:31 2023

@author: akhil_kk
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('imageY_1.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('imageY_2.jpg',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)

print(matches[0].queryIdx)
print(matches[0].trainIdx)

good_p1=[]
good_p2=[]

for item in matches[:10]:
    good_p1.append(kp1[item.queryIdx].pt)
    good_p2.append(kp2[item.queryIdx].pt)

print(good_p1)
print(good_p2)

print(np.asarray(good_p1))

#print(matches[:10])
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.





img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()



