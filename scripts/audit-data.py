import os
import cv2
import h5py
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt

def audit_image(img_filename):
    # 1. Résolution des chemins
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', 'data', 'ShanghaiTech_Dataset', 'part_A_final', 'train_data')
    
    img_path = os.path.join(base_dir, 'images', img_filename)
    mat_path = os.path.join(base_dir, 'ground_truth', img_filename.replace('IMG_', 'GT_IMG_').replace('.jpg', '.mat'))
    h5_path = os.path.join(base_dir, 'images', img_filename.replace('.jpg', '.h5'))

    # 2. Vérification mathématique stricte
    # A. Lecture des clics humains originaux
    mat = io.loadmat(mat_path)
    points = mat['image_info'][0, 0][0, 0][0]
    nombre_vrais_clics = len(points)

    # B. Lecture de notre génération H5
    with h5py.File(h5_path, 'r') as hf:
        density_map = hf['density'][:]
    masse_thermique = np.sum(density_map)

    print(f"--- Audit de {img_filename} ---")
    print(f"Clics humains exacts (.mat) : {nombre_vrais_clics}")
    print(f"Somme de notre carte (.h5)  : {masse_thermique:.2f}")
    print(f"Marge d'erreur de calcul    : {abs(nombre_vrais_clics - masse_thermique):.4f} personnes")

    # 3. Visualisation (La preuve par l'image)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"Image Originale")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(density_map, cmap='jet')
    plt.title(f"Carte de Densité pure ({masse_thermique:.0f} personnes)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(density_map, cmap='jet', alpha=0.5) # alpha=0.5 rend la carte semi-transparente
    plt.title("Superposition")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # On audite ton image précise
    audit_image('IMG_10.jpg')