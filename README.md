# DJI P4 Multispectral Image Processing

Preprocessing of images from DJI P4 Multispectral drone. 

- Lens distortion correction
- Bands alignment with edge detector and ECC Maximization

## Flags

- ```--source_folder```: folder with the data to be processed. If it's a subfolder, insert the path just with spaces (ex. path to folder instead of path/to/folder)
- ```--dest_folder```: folder where to save final data
- ```--save_npy```: store_true. Save data as numpy array (.npy format, float32)
- ```--save_jpg```: store_true. Save data as images (.jpg format)
- ```--edge```: Sobel or Canny edge detector
- ```--scale_factor```: Scale image to speed up the process
- ```--equalization```: store_true. Histogram equalization before saving jpg
- ```--veg_index```: store_true. Compute NDVI, GNDVI, and EVI

## Usage

```python3 process_images.py --source_folder path to folder --dest_folder path to folder --save_npy --save_jpg --edge sobel --scale_factor 0.8 --equalization --veg_index```
