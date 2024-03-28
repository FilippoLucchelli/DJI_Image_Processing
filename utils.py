import os
import numpy as np
import cv2
from collections import namedtuple
import time

Label=namedtuple('Label',
                 ['name', 'id', 'color'])

labels=[Label('grass', 0, (0,255,0)),
        Label('obstacle', 1, (183,50,250)),
        Label('road', 2, (151,151,151)),
        Label('trash', 3, (55,250,250)),
        Label('vegetation', 4, (11,120,11)),
        Label('sky', 5, (255,255,0))]

color2label={label.color: label for label in labels}

def cvatMask2np(mask):
    tolerance=30
    arr=np.zeros(mask.shape[:2])

    for label, color in enumerate(color2label.keys()):
        color=np.array(color)
        if label < len(color2label.keys()):
            arr[np.all(mask == color, axis=-1)] = label
    
    return arr.astype(np.uint8)
  

def get_paths(folder, format=''):
    file_names=sorted(os.listdir(folder))
    file_paths=[os.path.join(folder, file_names[i]) for i in range(len(file_names)) if file_names[i].endswith(format)]
    return file_paths

def divide_frames(paths, n_frames):
    paths_np=np.array(paths)
    reshaped_paths=paths_np.reshape(-1, n_frames)
    return reshaped_paths

def get_translation(meta):
    RelCenterX, RelCenterY = meta.get_item("XMP:RelativeOpticalCenterX"), meta.get_item("XMP:RelativeOpticalCenterY")
    return RelCenterX, RelCenterY

def get_gradient(img):
    grad_x=cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y=cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad=cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)

    return grad

def get_corners(img):
    """ Get corners of the black borders """
    mask_original=((img!=0)*255).astype(np.uint8) #create mask
    mask=cv2.copyMakeBorder(mask_original, 50, 50, 50, 50, cv2.BORDER_CONSTANT, None, value=0)

    dst=cv2.cornerHarris(mask, 5, 3, 0.04)
    ret, dst=cv2.threshold(dst, 0.1*dst.max(), 255, 0)
    dst=np.uint8(dst)
    _, _, _, centroids=cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(mask, np.float32(centroids), (5,5), (-1,-1), criteria)
    
    for corner in corners:
        for i in range(2):
            corner[i]=int(corner[i]-50)


    sorted_corners=sort_corners(np.uint16(corners[1:]))
    
    if len(sorted_corners==4):
        return sorted_corners
    else:
        print(f'{len(sort_corners)} corners found.')
        return None
    
def sort_corners(corners):
    """ Sorts corners in clockwise order based on angle """
    center_x = np.mean(corners[:, 0])
    center_y = np.mean(corners[:, 1])

    # Calculate angle for each corner
    angles = np.arctan2(corners[:, 1] - center_y, corners[:, 0] - center_x)

    # Sort corners based on angles (handling quadrant transitions)
    sorted_indices = np.argsort(angles + (angles < 0) * 2 * np.pi)
    sorted_corners = corners[sorted_indices]

    return sorted_corners

def get_rectangle(frame):
    _corners=[]
    shapes={'x_0': 0.0,
            'x_1': 0.0,
            'y_0': 0.0,
            'y_1': 0.0,
            'img_number': '',
            'width': 0,
            'height': 0}
    for path in frame:        
        img=np.load(path)
        _corners.append(get_corners(img))

    shapes['img_number']=os.path.basename(path)
    height, width=img.shape
    shapes['height'], shapes['width']= img.shape

    corners=np.stack(_corners, 0)    
    shapes['x_0']=np.max(corners[:,:, 0][corners[:,:, 0]<np.mean(corners[:,:, 0])])/width
    shapes['x_1']=np.min(corners[:,:, 0][corners[:,:, 0]>np.mean(corners[:,:, 0])])/width
    shapes['y_0']=np.max(corners[:,:, 1][corners[:,:, 1]<np.mean(corners[:,:, 1])])/height
    shapes['y_1']=np.min(corners[:,:, 1][corners[:,:, 1]>np.mean(corners[:,:, 1])])/height
    return shapes

