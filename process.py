import cv2
import numpy as np
import utils

def undistort(meta, img):
    
    CenterX=meta.get_item("XMP:CalibratedOpticalCenterX")
    CenterY=meta.get_item("XMP:CalibratedOpticalCenterY")
    DewarpData_string=meta.get_item("XMP:DewarpData")
    DD=[float(DewarpData_string.split(';')[1].split(',')[i]) for i in range(9)]
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = DD[0], DD[1], DD[2], DD[3], DD[4], DD[5], DD[6], DD[7], DD[8]
    CameraMatrix=np.array([[fx, 0, CenterX+cx], [0, fy, CenterY+cy], [0, 0, 1]])
    DistCoeff=np.array([k1, k2, p1, p2, k3])

    undistorted_img=cv2.undistort(img, CameraMatrix, DistCoeff)

    return undistorted_img


def align_bands_ECC(band, nir, meta, info, canny=False):

    if canny==True:
        band_edges=cv2.Canny(band, 100, 200)
        nir_edges=cv2.Canny(nir, 100, 200)

    else:
        band_smooth=cv2.GaussianBlur(band, (5,5), 0)
        nir_smooth=cv2.GaussianBlur(nir, (5,5), 0)
        band_edges=utils.get_gradient(band_smooth)
        nir_edges=utils.get_gradient(nir_smooth)

    warp_mode=cv2.MOTION_HOMOGRAPHY
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-8)

    x_shift, y_shift = utils.get_translation(meta)

    warp_matrix=np.array([[1.0, 0, x_shift],
                          [0, 1.0, y_shift],
                          [0, 0, 1.0]], dtype=np.float32)

    try:
        (cc, warp_matrix)=cv2.findTransformECC(nir_edges, band_edges, warp_matrix, warp_mode, criteria)

    except:
        print(f'Image number: {info[1]}, Band: {info[0]}')
        print('Warning: find transform failed.')

    aligned_band=cv2.warpPerspective(band, warp_matrix, (band.shape[1], band.shape[0]), flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
    return aligned_band
   

def align_bands_feat(band, nir, meta, info):
    akaze=cv2.AKAZE_create()
    bf=cv2.BFMatcher()

    kp_band, des_band=akaze.detectAndCompute(band, None)
    kp_nir, des_nir= akaze.detectAndCompute(nir, None)

    #print(type(kp_band))

    matches=bf.knnMatch(des_nir, des_band, k=2)
    print(len(matches))
    good_matches=[]

    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])
    print(len(good_matches))

    #print(good_matches)
    ref_matched_kpts=np.float32([kp_nir[m[0].queryIdx].pt for m in good_matches])
    sensed_matched_kpts=np.float32([kp_band[m[0].queryIdx].pt for m in good_matches])

    H, status=cv2.findHomography(sensed_matched_kpts, ref_matched_kpts, cv2.RANSAC, 5.0)
    warped_image=cv2.warpPerspective(band, H, (band.shape[1], band.shape[0]))
    del kp_band, des_band, kp_nir, des_nir, matches, good_matches, ref_matched_kpts, sensed_matched_kpts

    return warped_image
    
"""  except:
        print(f'Image number: {info[1]}, Band: {info[0]}')
        print('Warning: find transform failed.')
        exit()
 """