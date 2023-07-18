import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

import metadata
import utils

def undistort(path, meta):
    """ Fix lens distortion 
    
    args:
        path(str): path to image
        meta(metadata.Metadata): Metadata object containing the metadata from the picture

    returns:
        np.array: undistorted image
    """
    raw_img=plt.imread(path)
    
    #Get metadata from the picture
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
    """ Compute reflectance values for the image"""
    Irradiance=meta.get_item("XMP:Irradiance")

    radiance=utils.raw_image_to_radiance(meta, image)
    reflectance=radiance/Irradiance

    return reflectance

class VegetationIndices:
    """ Class to compute vegetation indices """
    def __init__(self, red, green, blue, nir):
        """ Initialize with bands """
        self.red=red
        self.green=green
        self.blue=blue
        self.nir=nir

    def compute_ndvi(self):
        """ Computation of Normalized Difference Vegetation Index """
        numerator = (self.nir - self.red)
        denominator = (self.nir + self.red)
        
        ndvi=np.zeros_like(denominator)
        
        mask=denominator > 0
        
        ndvi[mask]=numerator[mask]/denominator[mask]

        return ndvi
    
    def compute_gndvi(self):
        """ Computation of Green Normalized Difference Vegetation Index """
        numerator = (self.nir - self.green)
        denominator = (self.nir + self.green)
        
        ndvi=np.zeros_like(denominator)
        
        mask=denominator > 0
        
        ndvi[mask]=numerator[mask]/denominator[mask]

        return ndvi
    
    def compute_evi(self):
        """ Computation of enhanced vegetation index """
        return 2.5*(self.nir-self.red)/(self.nir+6*self.red-7.5*self.blue+1)
    

def translate_images(img, meta):
    """ Balance the offset given from different positions of camera """
    RelCenterX, RelCenterY = meta.get_item("XMP:RelativeOpticalCenterX"), meta.get_item("XMP:RelativeOpticalCenterY")
    
    M=np.array([[1, 0, RelCenterX], [0, 1, RelCenterY]])
    trans_img=cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return trans_img

def get_gradient(img):
    """ Get image gradient to compute ECC """
    grad_x=cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y=cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    grad=cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad.astype(np.float32)

def align_images(band, nir):
    """ Align 2 images using ECC algorithm """
    grad_band=get_gradient(band)
    grad_nir=get_gradient(nir)

    warp_mode=cv2.MOTION_HOMOGRAPHY
    warp_matrix=np.eye(3, 3, dtype=np.float32)

    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.0001)

    try:
        (cc, warp_matrix)=cv2.findTransformECC(grad_nir, grad_band, warp_matrix, warp_mode, criteria)

    except:
        print("Warning: find transform failed. Using identity as warp")
    
    aligned_img=cv2.warpPerspective(band, warp_matrix, (band.shape[1], band.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return aligned_img

def align_bands(imgs):
    """ Align all the different bands to NIR """
    blue=align_images(imgs[0], imgs[4])
    green=align_images(imgs[1], imgs[4])
    red=align_images(imgs[2], imgs[4])
    rededge=align_images(imgs[3], imgs[4])

    return blue, green, red, rededge, imgs[4]