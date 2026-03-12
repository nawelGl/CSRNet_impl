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

def process_dataset(root_path):
    """
    Parcourt l'arborescence, extrait les points des .mat et sauvegarde en .h5.
    """
    parts = ['part_A_final', 'part_B_final']
    splits = ['train_data', 'test_data']

    for part in parts:
        for split in splits:
            folder_path = os.path.join(root_path, part, split)
            img_path_pattern = os.path.join(folder_path, 'images', '*.jpg')
            
            img_paths = glob.glob(img_path_pattern)
            print(f"Traitement de {len(img_paths)} images dans {part}/{split}...")

            for img_path in img_paths:
                # 1. Lecture des dimensions de l'image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_shape = (img.shape[0], img.shape[1])

                # 2. Construction du chemin vers le .mat correspondant
                # Ex: .../images/IMG_1.jpg -> .../ground_truth/GT_IMG_1.mat
                mat_path = img_path.replace('images', 'ground_truth').replace('.jpg', '.mat').replace('IMG_', 'GT_IMG_')
                
                # 3. Extraction des coordonnées
                mat = io.loadmat(mat_path)
                points = mat['image_info'][0, 0][0, 0][0]

                # 4. Application du filtre selon la partie du dataset
                if part == 'part_A_final':
                    # Foule très dense : filtre adaptatif
                    density_map = generate_density_map_adaptive(img_shape, points)
                else:
                    # Foule éparse (Part B) : filtre fixe standard (sigma=15) comme décrit dans le papier CSRNet
                    density_map = np.zeros(img_shape, dtype=np.float32)
                    for pt in points:
                        x, y = min(int(pt[0]), img_shape[1]-1), min(int(pt[1]), img_shape[0]-1)
                        density_map[y, x] = 1.0
                    density_map = gaussian_filter(density_map, sigma=15)

                # 5. Sauvegarde optimale en HDF5 (.h5)
                h5_path = img_path.replace('.jpg', '.h5')
                with h5py.File(h5_path, 'w') as hf:
                    hf['density'] = density_map

if __name__ == '__main__':
    # Résolution automatique du chemin : on part du script, on remonte (..), on va dans data/ShanghaiTech_Dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(current_dir, '..', 'data', 'ShanghaiTech_Dataset')
    
    print(f"Dossier cible : {dataset_root}")
    process_dataset(dataset_root)
    print("Génération des cartes HDF5 terminée avec succès !")