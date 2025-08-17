import cv2
import numpy as np
from matplotlib import pyplot as plt
from CustomCalibrateCamera.Stereo_Calib_Camera import stereoCalibrateCamera
from CustomCalibrateCamera.Stereo_Calib_Camera import getStereoCameraParameters, getgetStereoSingleCameraParameters

import Camera.jetsonCam as jetCam

def draw_box(img,box):
   return cv2.rectangle(img,(box[0],box[1]) ,( box[0]+box[2], box[1]+box[3]), (255,0,0),2)

def crop_img(img,box):
   return img[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]


def common_roi(box1,box2):
    c_w=  int((box1[2]+box2[2])/2)
    c_h = int((box1[3]+box2[3])/2 )

    return  (box1[0],box1[1],c_w,c_h) , (box2[0],box2[1],c_w,c_h)

def get_combined_roi(roi1,roi2):
    x = min(roi1[0],roi2[0]) 
    y = min(roi1[1],roi2[1])
    w = max(roi1[2],roi2[2])
    h = max(roi1[3],roi2[3])
    return(x,y,w,h)

def crop_image(roi,img):
   return img[roi[0]:roi[0]+roi[2],roi[1]:roi[3],:]


def get_masked_img(img,mask):
   b = img[:,:,0]
   g = img[:,:,1]
   r = img[:,:,2]


def draw_lines(img):
   for i in range(0,img.shape[0],30):
       cv2.line(img,(0,i),(img.shape[1],i),(255,0,0),1)
   return img   

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
#stereoCalibrateCamera(cam1,cam2,'CustomCalibrateCamera/jetson_stereo_8MP',24)
lod_data = getStereoCameraParameters('CustomCalibrateCamera/jetson_stereo_8MP.npz')
lod_datac1 = getgetStereoSingleCameraParameters('CustomCalibrateCamera/jetson_stereo_8MPc1.npz')
lod_datac2 = getgetStereoSingleCameraParameters('CustomCalibrateCamera/jetson_stereo_8MPc2.npz')

print(lod_datac1[0])
print(lod_datac2[0])
print(lod_datac1[1])
print(lod_datac2[1])

camera_matrix_left = lod_data[0]
dist_coeffs_left =  lod_data[1]
camera_matrix_right =  lod_data[2]
dist_coeffs_right =  lod_data[3]
R =  lod_data[4]
T =  lod_data[5]
#print camera matrix
print("RAW camera matrix")
print(camera_matrix_right)
print(camera_matrix_left)


ret,img_s = cam1.read()
if ret:
  image_size = (img_s.shape[1],img_s.shape[0])
  print(image_size)
  
else:
  cam1.stop()
  cam2.stop()
  cam1.release()
  cam2.release()
  exit()

#stereo rectify
R1,R2,P1,P2,Q,roi1,roi2= cv2.stereoRectify(camera_matrix_left,dist_coeffs_left, camera_matrix_right, dist_coeffs_right, image_size, R, T)

block_s = 5
num_disp= 16

# Create a StereoBM object
stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_s)




# Load rectification maps
map1_left, map2_left = cv2.initUndistortRectifyMap( camera_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_16SC2)
map1_right, map2_right = cv2.initUndistortRectifyMap(camera_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_16SC2)

while True:
   # Read stereo images
   ret,image_left = cam1.read()
   ret, image_right = cam2.read()
   
   #gt optimal camera matrix and roi to compensate distortion
   #new_camera_matrix_left, roi1 = cv2.getOptimalNewCameraMatrix(lod_datac1[0], lod_datac1[1], image_size, 1)
   #new_camera_matrix_right, roi2 = cv2.getOptimalNewCameraMatrix(lod_datac2[0], lod_datac2[1], image_size, 1)

   #undistort images
   undistorted_img_left = cv2.undistort(image_left, camera_matrix_left, dist_coeffs_left) 
   undistorted_img_right = cv2.undistort(image_right, camera_matrix_right, dist_coeffs_right)

   # Remap the images using rectification maps
   rectified_left = cv2.remap(undistorted_img_left, map1_left, map2_left, cv2.INTER_LINEAR)
   rectified_right = cv2.remap(undistorted_img_right, map1_right, map2_right, cv2.INTER_LINEAR)

    # Convert images to grayscale
   gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
   gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

   # Compute disparity map
   disparity = stereo.compute(gray_left, gray_right)

   #disparity = disparity/16
   # Normalize the disparity map to the range [0, 1]
   normalized_disparity_map = cv2.normalize(disparity, None, 0.0, 1.0, cv2.NORM_MINMAX,cv2.CV_32F)
   #abs_disparity_map = np.abs(disparity)
   #normalized_disparity_map = cv2.normalize(abs_disparity_map, None, 0.0, 1.0, cv2.NORM_MINMAX,cv2.CV_32F)
   
   #undist_left  = draw_box(undistorted_img_left,roi1)
   #undist_right = draw_box(undistorted_img_right,roi2)
   
   baseline = np.linalg.norm(T)  # Magnitude of translation vector
   focal_length = camera_matrix_left[0, 0]  # Focal length (assuming both cameras have the same) in pixels

   baseline = 0.07  # Magnitude of translation vector
   focal_length =  0.00304
   depth_map =  focal_length * (baseline / (normalized_disparity_map + 1e-5)) 
   #print("min"+str(np.min(depth_map))+"  max:"+str(np.max(depth_map)))

   # Normalize the depth map to a specific range for visualization
   depth_map *=10000
   normalized_depth_map = cv2.normalize(depth_map, None, 0.0, 255, cv2.NORM_MINMAX,cv2.CV_32F) 
   
   #normalized_depth_map = 255 - normalized_depth_map
   #roi1, roi2  =common_roi(roi1,roi2)
   #undist_left  = crop_img(image_left,roi1)
   #undist_right = crop_img(image_right,roi2)
   depth_contrast = ((normalized_depth_map)).astype(np.uint8)
   #print(depth_contrast[400:450,300:350])
   #rundistorted image
   #cv2.imshow('undist img1', undistorted_img_left)   
   #cv2.imshow('undist img2', undistorted_img_right)   
   #cv2.imshow('undist img draw', cv2.hconcat([undist_left,undist_right])) 
   #cv2.imshow('undist img1 rect map', rectified_left) 
   #cv2.imshow('undist img2 rect map', rectified_right) 
   #cv2.imshow('Disparity', (normalized_disparity_map*255).astype(np.uint8))
   
   blur_disparity =filtered_disparity = cv2.GaussianBlur(normalized_disparity_map, (5, 5), 0)

   colormap_image = cv2.applyColorMap(np.uint8(blur_disparity * 255), cv2.COLORMAP_JET)
   cv2.imshow('Depth colour map',colormap_image )
   cv2.imshow('Depth',depth_contrast )
   image_left[:,:,2] = depth_contrast
   #print(depth_contrast[400:450,300:350])
   cv2.imshow('final',image_left)
   com_img=  cv2.hconcat([gray_left,gray_right])
   com_img=draw_lines(com_img)
   cv2.imshow('img_tog',com_img) 
   k=cv2.waitKey(10)
   if k == ord('x'):
     break
   elif k == ord('q'):
     #increase block size
     block_s +=2
     print("block_size:"+str(block_s))
     stereo.setBlockSize(block_s)

   elif k == ord('a'):
     #decrease block size
     block_s =max(block_s-2,5)
     print("block_size:"+str(block_s))
     stereo.setBlockSize(block_s)

   elif k == ord('w'):
     #increase disparity
     num_disp +=16
     print("disparity:"+str(num_disp))
     stereo.setNumDisparities(num_disp)

   elif k == ord('s'):
     #decrease disparity
     num_disp =max(16, num_disp-16)
     print("disparity:"+str(num_disp))
     stereo.setNumDisparities(num_disp)

cv2.destroyAllWindows()
cam1.stop()
cam2.stop()
cam1.release()
cam2.release()


