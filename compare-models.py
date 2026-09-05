import os
import torch
import cv2
import h5py
import math
import numpy as np
from torchvision import transforms
from model import CSRNet
from codecarbon import EmissionsTracker

def evaluer_modele(chemin_modele, img_paths, device, transform):
    print(f"\n Évaluation en cours pour : {chemin_modele}")
    model = CSRNet().to(device)
    
    # Sécurité si le modèle de base n'existe pas en fichier (ex: architecture vierge)
    if chemin_modele == "vierge":
        print("   (Utilisation des poids VGG-16 de base sans entraînement Part B)")
    else:
        model.load_state_dict(torch.load(chemin_modele, map_location=device))
        
    model.eval()
    
    erreur_absolue_totale = 0.0
    erreur_quadratique_totale = 0.0
    nombre_images = len(img_paths)
    
    with torch.no_grad():
        tracker = EmissionsTracker(project_name="CSRNet_Baseline_Mac")
        tracker.start()

        # enumerate permet de compter les images (i) en même temps qu'on les lit
        for i, img_path in enumerate(img_paths):
            h5_path = img_path.replace('.jpg', '.h5')
            with h5py.File(h5_path, 'r') as hf:
                compte_reel = np.sum(hf['density'][:])
                
            img_or = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img_tensor = transform(img_or).unsqueeze(0).to(device)
            
            compte_ia = np.sum(model(img_tensor).squeeze().cpu().numpy())
            
            erreur = compte_reel - compte_ia
            erreur_absolue_totale += abs(erreur)
            erreur_quadratique_totale += erreur ** 2
            
            # --- LOGS DE PROGRESSION (DANS la boucle) ---
            if (i + 1) % 50 == 0 or (i + 1) == nombre_images:
                print(f"   -> Progression : [{i+1}/{nombre_images}] images analysées...")

        # --- FIN DE LA BOUCLE FOR ---
        
        # 1. On stoppe le tracker une fois toutes les images traitées
        tracker.stop()

        # 2. On récupère et calcule l'énergie (HORS de la boucle)
        energie_kwh = tracker.final_emissions_data.energy_consumed # Énergie en kWh
        energie_joules = energie_kwh * 3.6e6 # Conversion en Joules

        print(f"\n⚡ BILAN ÉNERGÉTIQUE :")
        print(f"Énergie totale : {energie_joules:.2f} Joules")
        print(f"Énergie par image : {energie_joules / nombre_images:.4f} Joules/frame")

    mae = erreur_absolue_totale / nombre_images
    mse = math.sqrt(erreur_quadratique_totale / nombre_images)
    
    print(f" MAE : {mae:.2f} personnes (Moyenne d'erreur classique)")
    print(f" MSE : {mse:.2f} personnes (Sensibilité aux grosses erreurs)")
    return mae, mse

def lancer_comparaison():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ Exécution sur : {device}")

    # Préparation des données de Test
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(current_dir, 'data', 'ShanghaiTech_Dataset', 'part_B_final', 'test_data', 'images')
    img_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jpg')]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f" Chargement de {len(img_paths)} images inédites...")

    # --- 1. TEST DU MODÈLE DE BASE (Époque 1) ---
    modele_base = 'models/csrnet_epoch_1.pth' 
    
    if os.path.exists(modele_base):
        print(f" Fichier historique trouvé. Évaluation de {modele_base}...")
        mae_base, mse_base = evaluer_modele(modele_base, img_paths, device, transform)
    else:
        print(f"\n Fichier {modele_base} introuvable dans le dossier.")
        print(" Déclenchement du Fallback automatique : Création d'une architecture vierge...")
        mae_base, mse_base = evaluer_modele("vierge", img_paths, device, transform)

    # --- 2. TEST DU MEILLEUR MODÈLE (Best Époque) ---
    modele_best = 'models/csrnet_best.pth'
    if os.path.exists(modele_best):
        mae_best, mse_best = evaluer_modele(modele_best, img_paths, device, transform)
    else:
        print(f"\n Le fichier {modele_best} est introuvable.")
        return

    # --- 3. LE BILAN ---
    print("\n --- BILAN DE L'ENTRAÎNEMENT --- ")
    print(f" Amélioration MAE : -{mae_base - mae_best:.2f} personnes d'erreur moyenne")
    print(f" Amélioration MSE : -{mse_base - mse_best:.2f} (Réduction drastique des aberrations)")

if __name__ == '__main__':
    lancer_comparaison()