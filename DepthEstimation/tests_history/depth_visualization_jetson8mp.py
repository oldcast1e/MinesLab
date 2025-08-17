import cv2
import numpy as np
from matplotlib import pyplot as plt
from CustomCalibrateCamera.Stereo_Calib_Camera import stereoCalibrateCamera
from CustomCalibrateCamera.Stereo_Calib_Camera import getStereoCameraParameters

import Camera.jetsonCam as jetCam

def draw_box(img,box):
   return cv2.rectangle(img,(box[0],box[1]) ,( box[0]+box[2], box[1]+box[3]), (255,0,0),2)


def get_combined_roi(roi1,roi2):
    x = min(roi1[0],roi2[0]) 
    y = min(roi1[1],roi2[1])
    w = max(roi1[2],roi2[2])
    h = max(roi1[3],roi2[3])
    return(x,y,w,h)

def crop_image(roi,img):
   return img[roi[0]:roi[0]+roi[2],roi[1]:roi[3],:]


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

print(roi1)
print(roi2)


# Load rectification maps
map1_left, map2_left = cv2.initUndistortRectifyMap( camera_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_16SC2)
map1_right, map2_right = cv2.initUndistortRectifyMap(camera_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_16SC2)

while True:
   # Read stereo images
   ret,image_left = cam1.read()
   ret, image_right = cam2.read()
   
   # Remap the images using rectification maps
   rectified_left = cv2.remap(image_left, map1_left, map2_left, cv2.INTER_LINEAR)
   rectified_right = cv2.remap(image_right, map1_right, map2_right, cv2.INTER_LINEAR)
   
   #rectified_roi = get_combined_roi(roi1,roi2)

   #rectified_left=crop_image(rectified_roi,rectified_left)
   #rectified_right=crop_image(rectified_roi,rectified_right)

   # Convert images to grayscale
   gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
   gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

   # Compute disparity map
   disparity = stereo.compute(gray_left, gray_right)
   
   # Normalize the disparity map to the range [0, 1]
   normalized_disparity_map = cv2.normalize(disparity, None, 0.0, 1.0, cv2.NORM_MINMAX,cv2.CV_32F)
   #print(normalized_disparity_map.dtype)
   # Invert the disparity map to get depth values
   #depth_map = 1.0 / (normalized_disparity_map + 0.001)  # Add a small value to avoid division by zero

   baseline = np.linalg.norm(T)  # Magnitude of translation vector
   focal_length = camera_matrix_left[0, 0]  # Focal length (assuming both cameras have the same) in pixels

   #baseline = 70  # Magnitude of translation vector
   focal_length = camera_matrix_left[0, 0]  # Focal length (assuming both cameras have the same) in pixels
   depth_map = (baseline * focal_length) / (normalized_disparity_map + 1e-5) 

   # Normalize the depth map to a specific range for visualization
   normalized_depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)


   # Apply the heatmap colormap to the normalized depth map
   heatmap_depth_map = cv2.applyColorMap(np.uint8(normalized_depth_map * 255), cv2.COLORMAP_JET)

   # Display or save the heatmap visualization
   cv2.imshow('Disparity', (normalized_disparity_map*255).astype(np.uint8))
   cv2.imshow('Depth Map Heatmap', heatmap_depth_map)
   cv2.imshow('Depth Map contrast', normalized_depth_map)
   rectified_left  = draw_box(rectified_left,roi1)
   rectified_right = draw_box(rectified_right,roi2)

   cv2.imshow('stereo img', cv2.hconcat([rectified_left,rectified_right]))
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


