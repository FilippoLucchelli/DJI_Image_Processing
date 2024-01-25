import os
import metadata
import processing
import cv2
import numpy as np
import argparse
import utils
import time
import process

parser=argparse.ArgumentParser()

parser.add_argument('--source_folder', nargs='+', help='Ex. --folder path to folder -> \\path\\to\\folder')
parser.add_argument('--dest_folder', nargs='+', help='Ex. --folder path to folder -> \\path\\to\\folder')
parser.add_argument('--save_npy', action='store_false', help='Save numpy array')
parser.add_argument('--save_jpg', action='store_false', help='Save jpg image')
parser.add_argument('--edge', default='sobel', choices=['sobel', 'canny'])
parser.add_argument('--method', default='edge', choices=['edge', 'features'])

args=parser.parse_args()

BLACK_LEVEL=4096

source_folder=''
dest_folder=''

canny=False

if args.edge=='canny':
    canny=True

for fold in args.source_folder:
    source_folder=os.path.join(source_folder, fold)

for fold in args.dest_folder:
    dest_folder=os.path.join(dest_folder, fold)

if not os.path.isdir(dest_folder):
    os.mkdir(dest_folder)

if args.save_npy:
    npy_folder=os.path.join(dest_folder, 'NPY')
    if not os.path.isdir(npy_folder):
        os.mkdir(npy_folder)

if args.save_jpg:
    jpg_folder=os.path.join(dest_folder, 'JPG')
    if not os.path.isdir(jpg_folder):
        os.mkdir(jpg_folder)

img_paths=utils.get_paths(source_folder)
reshaped_paths=utils.divide_frames(img_paths)

##blue:1 green:2 red:3 rededge:4 nir:5

band_names={1: 'blue',
           2: 'green',
           3: 'red',
           4: 'rededge',
           5: 'nir'}
bands={}
undistorted_bands={}
metas={}
aligned={}
aligned_norm={}



for n_img, paths in enumerate(reshaped_paths):
    for n, path in enumerate(paths):
        bands[band_names[n+1]]=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        metas[band_names[n+1]]=metadata.Metadata(path)
        undistorted_bands[band_names[n+1]]=process.undistort(metas[band_names[n+1]], bands[band_names[n+1]])
    for key in undistorted_bands:
        if key == 'nir':
            aligned[key]=undistorted_bands['nir']

        else:
            if args.method=='edge':
                aligned[key]=process.align_bands_ECC(undistorted_bands[key], undistorted_bands['nir'], metas[key], info=(key, n_img), canny=canny)

            elif args.method=='features':
                aligned[key]=process.align_bands_feat(undistorted_bands[key], undistorted_bands['nir'], metas[key], info=(key, n_img))
        
        aligned[key][aligned[key]>BLACK_LEVEL]-=BLACK_LEVEL
        aligned_norm[key]=(aligned[key].astype(np.float32))/65535

        if args.save_npy:
            np.save(os.path.join(npy_folder, f'img_{n_img:03d}_{key}'), aligned_norm[key])

        if args.save_jpg:
            aligned[key]=cv2.equalizeHist(aligned[key])
            cv2.imwrite(os.path.join(jpg_folder, f'img_{n_img:03d}_{key}.jpg'), aligned[key])






