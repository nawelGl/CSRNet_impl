import os
import glob
import re
import random
import torch
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from model import CSRNet

def test_random_image(dataset_part='part_B_final'):
    # --- 1. CONFIGURATION MATÉRIELLE ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ Exécution sur : {device}")

    # --- 2. RECHERCHE AUTOMATIQUE DU MEILLEUR MODÈLE ---
    model = CSRNet().to(device)
    fichiers_modeles = glob.glob('models/csrnet_epoch_*.pth')
    
    if not fichiers_modeles:
        print("❌ Aucun modèle trouvé dans le dossier 'models/'.")
        return
        
    dernier_modele = max(fichiers_modeles, key=lambda f: int(re.search(r'epoch_(\d+)', f).group(1)))
    model.load_state_dict(torch.load(dernier_modele, map_location=device))
    
    model.eval() 
    print(f"🧠 Modèle chargé ({dernier_modele}) et verrouillé en mode Inférence.")

    # --- 3. SÉLECTION D'UNE IMAGE INCONNUE (Test Data) ---
    # MISE À JOUR : Utilisation dynamique de la variable dataset_part (Part B par défaut)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(current_dir, 'data', 'ShanghaiTech_Dataset', dataset_part, 'test_data')
    
    images_dir = os.path.join(test_dir, 'images')
    if not os.path.exists(images_dir):
        print(f"❌ Erreur : Le dossier {images_dir} n'existe pas. As-tu bien extrait la Part B ?")
        return

    img_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')]
    img_path = random.choice(img_paths)
    nom_image = os.path.basename(img_path)
    
    # Correction robuste du chemin du corrigé
    h5_path = img_path.replace('.jpg', '.h5') 

    print(f"\n📸 Test sur l'image inédite : {nom_image} (Dataset : {dataset_part})")

    # --- 4. PRÉPARATION DE L'IMAGE ---
    img_or = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_or).unsqueeze(0).to(device)

    # --- 5. LA PRÉDICTION DE L'IA ---
    with torch.no_grad():
        output = model(img_tensor)
    
    pred_map = output.squeeze().cpu().numpy()
    compte_ia = np.sum(pred_map)

    # --- 6. RÉCUPÉRATION DE LA VÉRITÉ TERRAIN ---
    with h5py.File(h5_path, 'r') as hf:
        gt_map = hf['density'][:]
    compte_reel = np.sum(gt_map)

    print(f"🎯 Vérité terrain (Humain) : {compte_reel:.0f} personnes")
    print(f"🤖 Prédiction de l'IA      : {compte_ia:.0f} personnes")
    print(f"📉 Marge d'erreur absolue  : {abs(compte_reel - compte_ia):.0f} personnes")

    # --- 7. VISUALISATION DES RÉSULTATS ---
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img_or)
    plt.title(f"Image Test : {nom_image}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gt_map, cmap='jet')
    plt.title(f"Corrigé Humain ({compte_reel:.0f} pers.)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_map, cmap='jet')
    plt.title(f"Prédiction IA ({compte_ia:.0f} pers.)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Tu peux changer ici par 'part_A_final' si tu veux re-tester l'ancien dataset un jour
    test_random_image(dataset_part='part_B_final')