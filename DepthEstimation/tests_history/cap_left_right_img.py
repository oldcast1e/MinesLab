#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 23:47:14 2023

@author: akhil_kk
"""

import cv2

import cv2
cap=cv2.VideoCapture(0)
i=0
while i<30:
    ret,frame=cap.read()
    if ret:
        if i==29:
           cv2.imwrite("imageY_2.jpg",frame)
        cv2.imshow("img",frame)
        cv2.waitKey(2)
        i+=1
cv2.destroyAllWindows()   
cap.release()