# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 19:52:23 2020

@author: AKHIL_KK
"""


import cv2
import urllib
import numpy as np
#import json

class IPCam():
    active_cameras = 0
    http = urllib
    #focus modes
    auto = 'auto'
    manual = 'off'
    macro = 'macro'
    smooth = 'continuous-video'
    aggressive = 'continuous-picture'
    
    def __init__(self,url):
        video = '/video'
#        save_af_photo = '/photoaf_save_only.jpg'
#        photo_af = '/photoaf.jpg'
#        photo = '/photo.jpg'

        self.url = url
        self.cap = cv2.VideoCapture(self.url+video)
        IPCam.active_cameras +=1
   
    def getFrames(self):
        return(self.cap.read())
        
    def getLatestFrame(self):
        pic = '/shot.jpg'
        url_response= (IPCam.http.request.urlopen(self.url +pic))
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)
        return img
    def setZoom(self, zoom_value,zoom_min = 0,zoom_max=100):
        #zoom_max= 100
        zoom_inc = 0.01
        zoom_cam = '/ptz?zoom='
        if zoom_value >= zoom_min and zoom_value <= zoom_max:
            IPCam.http.request.urlopen( self.url + zoom_cam + str(zoom_value))
        elif zoom_value < zoom_min:
            return zoom_value - zoom_min
        else:
            return zoom_value - zoom_max
        return 0
    
    def setQuality(self, quality_val,qual_min = 1,qual_max =100):
        #qual_max =100
        qual_inc = 1
        qual_cam = '/settings/quality?set='
        if quality_val >= qual_min and quality_val <= qual_max:
            IPCam.http.request.urlopen( self.url + qual_cam + str(quality_val))
        elif quality_val < qual_min:
            return quality_val - qual_min
        else:
            return quality_val - qual_max
        return 0
    

    def A_setExposure(self, exp_val,expo_min = 10783,expo_max = 31617300000):
        #expo_max = 12
        expo_inc = 1
        expo_cam = '/settings/exposure_ns?set='       
        expo_mode= '/settings/manual_sensor?set=on'
        IPCam.http.request.urlopen( self.url + expo_mode ) #setting expo mode to manula first
        
        if exp_val >= expo_min and exp_val <= expo_max:
            IPCam.http.request.urlopen( self.url + expo_cam + str(exp_val))
        elif exp_val < expo_min:
            return exp_val - expo_min
        else:
            return exp_val - expo_max
        return 0
    
    def flashOn(self):
        torch_on = '/enabletorch'
        IPCam.http.request.urlopen( self.url + torch_on)
    
    def flashOff(self):
        torch_off = '/disabletorch'
        IPCam.http.request.urlopen( self.url + torch_off)
    
    def focus(self):
        focus = '/focus'
        IPCam.http.request.urlopen( self.url + focus)
    
    def A_setFocus(self,focus_val,focus_min = 0,focus_max=13):
        focus_inc = 0.01
        focus_mode= '/settings/focusmode?set='+IPCam.manual
        focus_cam = '/settings/focus_distance?set='
        IPCam.http.request.urlopen( self.url + focus_mode ) #setting focus mode to manula first
        
        if focus_val >= focus_min and focus_val <= focus_max:
            IPCam.http.request.urlopen( self.url + focus_cam + str(focus_val))
        elif focus_val < focus_min:
            return focus_val - focus_min
        else:
            return focus_val - focus_max
        return 0
    
    def A_setFocusMode(self,focus_mode = auto):
        if focus_mode==IPCam.auto or focus_mode==IPCam.manual or \
        focus_mode==IPCam.macro or focus_mode==IPCam.smooth or focus_mode==IPCam.aggressive: 
           focus_mode= '/settings/focusmode?set='+focus_mode
           IPCam.http.request.urlopen( self.url + focus_mode ) #setting focus mode
           return 0
        else:
            return 1   #return 1 if invalid focus mode requested
           
    def noFocus(self):
        nofocus = '/nofocus'
        IPCam.http.request.urlopen( self.url + nofocus)
        return 0
        
    
    
    def A_setSensorControl(self,sensor_mode=auto):
        auto_cam='/settings/manual_sensor?set=off'
        manual_cam='/settings/manual_sensor?set=on'
        if sensor_mode==IPCam.auto:
            IPCam.http.request.urlopen( self.url + auto_cam)
        elif sensor_mode==IPCam.manual:
            IPCam.http.request.urlopen( self.url + manual_cam)
    
    def A_setIso(self,iso_val,iso_min = 100,iso_max=5699):
        iso_inc = 1
        iso_mode= '/settings/manual_sensor?set=on'
        iso_cam = '/settings/iso?set='
        IPCam.http.request.urlopen( self.url + iso_mode ) #setting iso mode to manula first
        
        if iso_val >= iso_min and iso_val <= iso_max:
            IPCam.http.request.urlopen( self.url + iso_cam + str(iso_val))
        elif iso_val < iso_min:
            return iso_val - iso_min
        else:
            return iso_val - iso_max
        return 0
    
    def A_setShutter(self,shutter_val,shutter_min = 33333333,shutter_max=31617425424):
        shutter_inc = 1
        shutter_mode= '/settings/manual_sensor?set=on'
        shutter_cam = '/settings/frame_duration?set='
        IPCam.http.request.urlopen( self.url + shutter_mode ) #setting shutter mode to manula first
        
        if shutter_val >= shutter_min and shutter_val <= shutter_max:
            IPCam.http.request.urlopen( self.url + shutter_cam + str(shutter_val))
        elif shutter_val < shutter_min:
            return shutter_val - shutter_min
        else:
            return shutter_val - shutter_max
        return 0

    def switchToFrontCam(self):
        fron_cam = '/settings/ffc?set=on'
        IPCam.http.request.urlopen( self.url + fron_cam)
        return 0
        
    def switchToRearCam(self):
        main_cam = '/settings/ffc?set=off'
        IPCam.http.request.urlopen( self.url + main_cam)
        return 0
     
    def sensorData(self):
        sensor = '/sensors.json'
        return(IPCam.http.request.urlopen( self.url + sensor))
    
    def __del__(self):
        self.cap.release()
        IPCam.active_cameras -=1
        return 0

  

   
