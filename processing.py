import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

import metadata
import utils

def undistort(path, meta):
    raw_img=plt.imread(path)
    
    BlackLevel=meta.get_item("EXIF:BlackLevel")
    VignettingData_string=meta.get_item("XMP:VignettingData")
    VD=[float(VignettingData_string.split(',')[i]) for i in range(6)]
    CenterX=meta.get_item("XMP:CalibratedOpticalCenterX")
    CenterY=meta.get_item("XMP:CalibratedOpticalCenterY")
    DewarpData_string=meta.get_item("XMP:DewarpData")
    DD=[float(DewarpData_string.split(';')[1].split(',')[i]) for i in range(9)]
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = DD[0], DD[1], DD[2], DD[3], DD[4], DD[5], DD[6], DD[7], DD[8]
    CameraMatrix=np.array([[fx, 0, CenterX+cx], [0, fy, CenterY+cy], [0, 0, 1]])
    DistCoeff=np.array([k1, k2, p1, p2, k3])
    ValGain=meta.get_item("XMP:SensorGain")
    ValETime=meta.get_item("XMP:ExposureTime")/1e6
    PCam=meta.get_item("XMP:SensorGainAdjustment")

    #norm_img=(raw_img-BlackLevel)/65535
    undistorted_img=cv2.undistort(raw_img, CameraMatrix, DistCoeff)
    

    return undistorted_img

def compute_reflectance(meta, image):
    Irradiance=meta.get_item("XMP:Irradiance")

    radiance=utils.raw_image_to_radiance(meta, image)
    reflectance=radiance/Irradiance

    return reflectance

class VegetationIndices:
    def __init__(self, red, green, blue, nir):
        self.red=red
        self.green=green
        self.blue=blue
        self.nir=nir

    def compute_ndvi(self):
        return (self.nir-self.red)/(self.nir+self.red)
    
    def compute_evi(self):
        return 2.5*(self.nir-self.red)/(self.nir+6*self.red-7.5*self.blue+1)
    
