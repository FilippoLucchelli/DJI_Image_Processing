import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

import metadata
import utils

def raw2reflectance(path):
    raw_img=plt.imread(path)
    meta=metadata.Metadata(path)

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
    Irradiance=meta.get_item("XMP:Irradiance")

    norm_img=(raw_img-BlackLevel)/65535
    undistorted_img=cv2.undistort(norm_img, CameraMatrix, DistCoeff)
    img_camera=undistorted_img/(ValGain*ValETime)
    img_ref=img_camera*(PCam/Irradiance)

    return img_ref, undistorted_img

def vignette_correction(norm_img, CenterX, CenterY, DD):
    corrected_img=np.zeros_like(norm_img)
    for x in range(norm_img.shape[1]):
        for y in range(norm_img.shape[0]):
            r=math.sqrt((x - CenterX)**2 + (y - CenterY)**2)
            corrected_img[y,x]=norm_img[y,x]*(DD[5]*(r**6) + DD[4]*(r**5) + DD[3]*(r**4) + DD[2]*(r**3) + DD[1]*(r**2) + DD[0]*(r**1) + 1)
    return corrected_img