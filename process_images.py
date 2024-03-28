import os
import metadata
import cv2
import numpy as np
import argparse
import utils
import process
import veg_index

parser=argparse.ArgumentParser()

parser.add_argument('--source_folder', nargs='+', help='Ex. --folder path to folder -> \\path\\to\\folder')
parser.add_argument('--dest_folder', nargs='+', help='Ex. --folder path to folder -> \\path\\to\\folder')
parser.add_argument('--save_npy', action='store_true', help='Save numpy array')
parser.add_argument('--save_jpg', action='store_true', help='Save jpg image')
parser.add_argument('--edge', default='sobel', choices=['sobel', 'canny'], help='Edge detection method. Sobel with gaussian filter or canny')
parser.add_argument('--scale_factor', default=1, type=float, help='Scale factor to speed-up processing. Final image is not resized. shape=(w*factor, h*factor)')
parser.add_argument('--equalization', action='store_true', help='Histogram equalization before saving jpg. For visualization only.')
parser.add_argument('--veg_index', action='store_true', help='Compute NDVI, GNDVI, and EVI')

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

img_paths=utils.get_paths(source_folder, 'TIF')
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
save=True



for n_img, paths in enumerate(reshaped_paths):
   
    for n, path in enumerate(paths):
        bands[band_names[n+1]]=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        metas[band_names[n+1]]=metadata.Metadata(path)
        undistorted_bands[band_names[n+1]]=process.undistort(metas[band_names[n+1]], bands[band_names[n+1]])

    for key in undistorted_bands:
        if os.path.isfile(os.path.join(npy_folder, f'img_{n_img:03d}_{key}.npy')):
            save=False
        
        else:
            save=True
            if key == 'nir':
                aligned[key]=undistorted_bands['nir']

            else:
                aligned[key]=process.align_bands_ECC(undistorted_bands[key], undistorted_bands['nir'], 
                                                    metas[key], info=(key, n_img), canny=canny, scale_factor=args.scale_factor)
        
            aligned[key][aligned[key]>BLACK_LEVEL]-=BLACK_LEVEL
            aligned_norm[key]=(aligned[key].astype(np.float32))/65535

            if args.save_npy:
                np.save(os.path.join(npy_folder, f'img_{n_img:03d}_{key}'), aligned_norm[key])

            if args.save_jpg:
                if args.equalization:
                    aligned[key]=cv2.equalizeHist(aligned[key])
                cv2.imwrite(os.path.join(jpg_folder, f'img_{n_img:03d}_{key}.jpg'), aligned[key])

    if save:    
        if args.veg_index:
            vis={}
            vi=veg_index.VegetationIndices(aligned_norm['red'], aligned_norm['green'], aligned_norm['blue'], aligned_norm['nir'])
            vis['ndvi']=vi.compute_ndvi()
            vis['evi']=vi.compute_evi()
            vis['gndvi']=vi.compute_gndvi()
        if args.save_npy:
            for key in vis:
                np.save(os.path.join(npy_folder, f'img_{n_img:03d}_{key}'), vis[key])
        if args.save_jpg:
            for key in vis:
                norm_vi=cv2.normalize(vis[key], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                if args.equalization:
                    norm_vi=cv2.equalizeHist(norm_vi)
                cv2.imwrite(os.path.join(jpg_folder, f'img_{n_img:03d}_{key}.jpg'), norm_vi)
                    




