import os
import glob
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ShanghaiTechDataset(Dataset):
    """
    Classe standard PyTorch pour lire les images et les corrigés à la volée.
    """
    def __init__(self, root_path, part='part_A_final', split='train_data'):
        # On liste tous les chemins vers les images .jpg
        self.img_paths = glob.glob(os.path.join(root_path, part, split, 'images', '*.jpg'))
        
        # BONNE PRATIQUE : Normalisation ImageNet
        # Comme notre modèle utilise un "cerveau" VGG-16 pré-entraîné sur ImageNet,
        # on doit lui donner des couleurs avec la même colorimétrie que ce qu'il connaît.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        """Retourne le nombre total d'images dans le dossier."""
        return len(self.img_paths)

    def __getitem__(self, index):
        """
        Fonction appelée automatiquement par PyTorch pendant l'entraînement.
        Elle pioche UNE image au hasard, la prépare, et la renvoie avec son corrigé.
        """
        img_path = self.img_paths[index]
        h5_path = img_path.replace('.jpg', '.h5')

        # 1. Lecture de l'image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV lit en BGR, PyTorch veut du RGB
        
        # 2. Lecture du corrigé (la matrice .h5 qu'on a générée)
        with h5py.File(h5_path, 'r') as hf:
            density_map = hf['density'][:]

        # 3. SOLUTION OPTIMALE : L'ajustement des dimensions
        # On a vu que le modèle CSRNet réduit la taille de l'image par 8.
        # On doit donc réduire le corrigé par 8 pour pouvoir les comparer.
        ht, wd = density_map.shape
        new_ht = ht // 8
        new_wd = wd // 8
        
        # ASTUCE MATHÉMATIQUE : Si on réduit la surface par 64 (8 en largeur x 8 en hauteur),
        # on perd de la "masse thermique". Pour que le compte des personnes reste exact,
        # on doit multiplier la valeur des pixels par 64 !
        density_map = cv2.resize(density_map, (new_wd, new_ht), interpolation=cv2.INTER_CUBIC) * 64.0
        
        # 4. Conversion en Tenseurs PyTorch (Le format que le modèle comprend)
        img_tensor = self.transform(img)
        density_tensor = torch.from_numpy(density_map).unsqueeze(0) # On ajoute un canal [1, H, W]

        return img_tensor, density_tensor

# --- Bloc de test rapide ---
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(current_dir, 'data', 'ShanghaiTech_Dataset')
    
    print(f"Recherche des données dans : {dataset_root}")
    dataset = ShanghaiTechDataset(root_path=dataset_root, part='part_A_final', split='train_data')
    
    print(f"Dataset chargé avec succès ! Nombre d'images trouvées : {len(dataset)}")
    
    if len(dataset) > 0:
        # NOUVEAU : On récupère le nom exact du fichier à l'index 0
        chemin_image = dataset.img_paths[0]
        nom_image = os.path.basename(chemin_image)
        
        img, target = dataset[0]
        
        print(f"\n--- Test sur l'image : {nom_image} ---")
        print(f"Taille du tenseur Image : {img.shape}")
        print(f"Taille du tenseur Corrigé : {target.shape}")
        print(f"Nombre de personnes exact sur cette image : {target.sum().item():.2f}")