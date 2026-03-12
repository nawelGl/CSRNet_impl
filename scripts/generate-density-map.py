import numpy as np
import scipy.io as io
from scipy.ndimage import gaussian_filter
import scipy.spatial
import h5py
import os
import glob
import cv2

def generate_density_map_adaptive(img_shape, points):
    """
    Source : https://github.com/leeyeehoo/CSRNet-pytorch/blob/master/make_dataset.ipynb?short_path=c8e7e9b
    ==> Code adapté à l'aide de Gemini
    Génère une carte adaptative pour la Part A (Foules très denses).
    Utilise KDTree pour ajuster la taille du flou en fonction de la distance des voisins.
    """
    density_map = np.zeros(img_shape, dtype=np.float32)
    num_pts = len(points)
    
    if num_pts == 0:
        return density_map

    for pt in points:
        x, y = min(int(pt[0]), img_shape[1]-1), min(int(pt[1]), img_shape[0]-1)
        density_map[y, x] = 1.0

    if num_pts < 4:
        return gaussian_filter(density_map, sigma=15)

    tree = scipy.spatial.KDTree(points)
    distances, _ = tree.query(points, k=4)
    d_bar = distances[:, 1:].mean(axis=1)
    
    density_map_final = np.zeros(img_shape, dtype=np.float32)
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        x, y = min(int(pt[0]), img_shape[1]-1), min(int(pt[1]), img_shape[0]-1)
        pt2d[y, x] = 1.0
        
        sigma = 0.3 * d_bar[i]
        density_map_final += gaussian_filter(pt2d, sigma=sigma)
        
    return density_map_final