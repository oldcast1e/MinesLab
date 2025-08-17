import open3d as o3d
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


def getPoints(points,img_disp,img):
   points[:,2]=img_disp.flatten()
   img=img/255
   colours = np.stack(((img[:,:,2]).flatten(),(img[:,:,1]).flatten(),(img[:,:,0]).flatten()),1)
   return points, colours

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

print('here4')
#stereo rectify
R1,R2,P1,P2,Q,roi1,roi2= cv2.stereoRectify(camera_matrix_left,dist_coeffs_left, camera_matrix_right, dist_coeffs_right, image_size, R, T)

block_s = 5
num_disp= 16

# Create a StereoBM object
#stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_s)
stereo = cv2.cuda.createStereoBM(numDisparities=num_disp, blockSize=block_s)


# Load rectification maps
map1_left, map2_left = cv2.initUndistortRectifyMap( camera_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_16SC2)
map1_right, map2_right = cv2.initUndistortRectifyMap(camera_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_16SC2)

# Initialize GPU capture object
gpu_mat_l = cv2.cuda_GpuMat()
gpu_mat_r = cv2.cuda_GpuMat()

# Create output disparity map
gpu_disparity = cv2.cuda_GpuMat()

# Create CUDA stream
stream = cv2.cuda_Stream()

h,w=image_size[:2]

total_pix=w*h
points = np.ones((total_pix,3)) *255
colours= np.ones((total_pix,3))
#points = np.random.rand(300, 3)



count = 0
for i in range(0,w,1):
  for j in range(0,h,1):
    if count ==total_pix:
      break
    points[count]= [i,j,0 ]
    count+=1

# Step 2: Convert the NumPy array to an Open3D PointCloud object
point_cloud = o3d.geometry.PointCloud()


point_cloud.points = o3d.utility.Vector3dVector(points)
#point_cloud.colors = o3d.utility.Vector3dVector(colours)
# Step 3: Visualize the PointCloud using Open3D's visualization tools
#o3d.visualization.draw_geometries([point_cloud])


import time

vis = o3d.visualization.Visualizer()
if vis is None:
  print('vis is none')
  cam1.stop()
  cam2.stop()
  cam1.release()
  cam2.release()
  exit()

vis.create_window(
    window_name='Carla Lidar',
    width=960,
    height=540,
    left=480,
    top=270)
vis.get_render_option().background_color = [0.05, 0.05, 0.05]
vis.get_render_option().point_size = 1
vis.get_render_option().show_coordinate_frame = True

vis.add_geometry(point_cloud)

frame = 0



while True:
   # Read stereo images
   ret,image_left = cam1.read()
   ret, image_right = cam2.read()

   # Remap the images using rectification maps
   rectified_left = cv2.remap(image_left, map1_left, map2_left, cv2.INTER_LINEAR)
   rectified_right = cv2.remap(image_right, map1_right, map2_right, cv2.INTER_LINEAR)

   gpu_mat_l.upload(rectified_left)
   gpu_mat_r.upload(rectified_right)

   # Convert images to grayscale
   gray_left = cv2.cuda.cvtColor(gpu_mat_l, cv2.COLOR_BGR2GRAY)
   gray_right = cv2.cuda.cvtColor(gpu_mat_r, cv2.COLOR_BGR2GRAY)

   # Compute disparity map
   gpu_disparity = stereo.compute(gray_left, gray_right, stream=stream)

   # Download disparity map from GPU to CPU memory
   disparity = gpu_disparity.download()

   # Normalize the disparity map to the range [0, 1]
   normalized_disparity_map = cv2.normalize(disparity, None, 0.0, 1.0, cv2.NORM_MINMAX,cv2.CV_32F)


   colormap_image = cv2.applyColorMap(np.uint8(normalized_disparity_map * 255), cv2.COLORMAP_JET)
   
   #print(time.time(),end=' ')
   points, colours= getPoints(points,normalized_disparity_map*255, colormap_image)
   #print(time.time(),)
   #points = np.random.rand(300, 3)
   point_cloud.points = o3d.utility.Vector3dVector(points)
   point_cloud.colors = o3d.utility.Vector3dVector(colours)
   vis.update_geometry(point_cloud)
   vis.poll_events()
   vis.update_renderer()
   # This can fix Open3D jittering issues:
   time.sleep(0.005)

   cv2.imshow('Depth colour map',colormap_image )
   #com_img=  cv2.hconcat([rectified_left,rectified_right])
   #com_img=draw_lines(com_img)
   #cv2.imshow('img_tog',com_img) 
   k=cv2.waitKey(20)
   
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


