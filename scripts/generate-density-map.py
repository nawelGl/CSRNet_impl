import numpy as np
import scipy.io as io
from scipy.ndimage import gaussian_filter
import scipy.spatial
import h5py
import os
import glob
import cv2

def generate_density_map(img_shape, points):
    """
    Source : https://github.com/leeyeehoo/CSRNet-pytorch/blob/master/make_dataset.ipynb?short_path=c8e7e9b
    Génère une carte de densité à partir des coordonnées des têtes.
    img_shape : tuple (hauteur, largeur) de l'image originale.
    points : tableau numpy contenant les coordonnées (X, Y) des têtes.
    """
    density_map = np.zeros(img_shape, dtype=np.float32)
    num_pts = len(points)
    
    # Cas extrême : si l'image est vide (aucune personne)
    if num_pts == 0:
        return density_map

    # On place un '1' aux coordonnées exactes de chaque tête
    for pt in points:
        # Attention: x et y peuvent être inversés selon le format du dataset
        x, y = min(int(pt[0]), img_shape[1]-1), min(int(pt[1]), img_shape[0]-1)
        density_map[y, x] = 1.0

    # S'il y a très peu de monde (ex: 1 à 3 personnes), un filtre fixe suffit
    if num_pts < 4:
        return gaussian_filter(density_map, sigma=15)

    # --- L'adaptatif (pour les foules denses) ---
    # On crée un arbre spatial pour trouver les voisins les plus proches instantanément
    tree = scipy.spatial.KDTree(points)
    
    # On trouve la distance des 4 voisins les plus proches pour chaque point
    # k=4 car le point lui-même compte (distance 0), donc on prend les 3 autres.
    distances, _ = tree.query(points, k=4)
    
    # On calcule la distance moyenne avec ces voisins
    d_bar = distances[:, 1:].mean(axis=1)
    
    # On crée la carte finale en additionnant les "taches" (flous gaussiens) de chaque point
    density_map_final = np.zeros(img_shape, dtype=np.float32)
    
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        x, y = min(int(pt[0]), img_shape[1]-1), min(int(pt[1]), img_shape[0]-1)
        pt2d[y, x] = 1.0
        
        # Le sigma (la taille de la tache) dépend de la distance aux voisins.
        # Plus les gens sont serrés, plus d_bar est petit, plus la tache est petite.
        # Le coefficient 0.3 est empirique (défini par les auteurs de CSRNet)
        sigma = 0.3 * d_bar[i]
        
        density_map_final += gaussian_filter(pt2d, sigma=sigma)
        
    return density_map_final