import os
import numpy as np
import cv2

def get_paths(folder):
    file_names=sorted(os.listdir(folder))
    file_paths=[os.path.join(folder, file_names[i]) for i in range(len(file_names))]
    return file_paths

def divide_frames(paths):
    paths_np=np.array(paths)
    reshaped_paths=paths_np.reshape(-1, 5)
    return reshaped_paths


def get_translation(meta):
    RelCenterX, RelCenterY = meta.get_item("XMP:RelativeOpticalCenterX"), meta.get_item("XMP:RelativeOpticalCenterY")
    return RelCenterX, RelCenterY

def get_gradient(img):
    grad_x=cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y=cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad=cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)

    return grad