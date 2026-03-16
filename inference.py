import os
import random
import torch
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from model import CSRNet

def test_random_image():
    # --- 1. CONFIGURATION MATÉRIELLE ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ Exécution sur : {device}")

    # --- 2. CHARGEMENT DU CERVEAU (Époque 10) ---
    model = CSRNet().to(device)
    chemin_modele = 'models/csrnet_epoch_10.pth'
    
    if not os.path.exists(chemin_modele):
        print("❌ Modèle introuvable. As-tu bien terminé l'entraînement ?")
        return
        
    model.load_state_dict(torch.load(chemin_modele, map_location=device))
    
    # BONNE PRATIQUE : Mode Évaluation
    # Désactive les comportements d'entraînement (comme le calcul des statistiques pour la BatchNorm)
    model.eval() 
    print("🧠 Modèle chargé et verrouillé en mode Inférence.")

    # --- 3. SÉLECTION D'UNE IMAGE INCONNUE (Test Data) ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(current_dir, 'data', 'ShanghaiTech_Dataset', 'part_A_final', 'test_data')
    
    images_dir = os.path.join(test_dir, 'images')
    img_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    img_path = random.choice(img_paths)
    nom_image = os.path.basename(img_path)
    h5_path = img_path.replace('.jpg', '.h5')

    print(f"\n📸 Test sur l'image inédite : {nom_image}")

    # --- 4. PRÉPARATION DE L'IMAGE ---
    # On applique EXACTEMENT la même normalisation que pendant l'entraînement
    img_or = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_or).unsqueeze(0).to(device) # Ajout de la dimension "batch"

    # --- 5. LA PRÉDICTION DE L'IA ---
    # BONNE PRATIQUE : Couper les gradients
    with torch.no_grad():
        output = model(img_tensor)
    
    # On récupère la carte thermique prédite et on calcule la somme
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

    # Image originale
    plt.subplot(1, 3, 1)
    plt.imshow(img_or)
    plt.title(f"Image Test : {nom_image}")
    plt.axis('off')

    # Corrigé (Ground Truth)
    plt.subplot(1, 3, 2)
    plt.imshow(gt_map, cmap='jet')
    plt.title(f"Corrigé Humain ({compte_reel:.0f} pers.)")
    plt.axis('off')

    # Prédiction de l'IA
    plt.subplot(1, 3, 3)
    plt.imshow(pred_map, cmap='jet')
    plt.title(f"Prédiction IA ({compte_ia:.0f} pers.)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_random_image()