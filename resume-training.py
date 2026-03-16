import os
import glob
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CSRNet
from dataset import ShanghaiTechDataset

def resume_training_auto(epochs_to_add=5):
    # --- 1. CONFIGURATION MATÉRIELLE ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Accélération matérielle Apple Silicon (MPS) activée !")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # --- 2. PRÉPARATION DES DONNÉES ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(current_dir, 'data', 'ShanghaiTech_Dataset')
    
    train_dataset = ShanghaiTechDataset(root_path=dataset_root, part='part_A_final', split='train_data')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # --- 3. INITIALISATION ET RECHERCHE AUTOMATIQUE ---
    model = CSRNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Recherche de tous les fichiers .pth dans le dossier
    fichiers_modeles = glob.glob('models/csrnet_epoch_*.pth')
    
    if not fichiers_modeles:
        print("❌ Aucun modèle trouvé dans le dossier 'models/'. Utilise d'abord train.py pour démarrer de zéro.")
        return

    # SOLUTION OPTIMALE : Extraction du plus grand numéro via Expression Régulière (Regex)
    dernier_modele = max(fichiers_modeles, key=lambda f: int(re.search(r'epoch_(\d+)', f).group(1)))
    derniere_epoque = int(re.search(r'epoch_(\d+)', dernier_modele).group(1))
    
    # Chargement des poids
    model.load_state_dict(torch.load(dernier_modele, map_location=device))
    print(f"🔄 Reprise automatique : Cerveau de l'époque {derniere_epoque} chargé depuis {dernier_modele}")

    # --- 4. LA BOUCLE D'APPRENTISSAGE DYNAMIQUE ---
    start_epoch = derniere_epoque + 1
    end_epoch = derniere_epoque + epochs_to_add
    
    print(f"\n🧠 Lancement de l'entraînement pour les époques {start_epoch} à {end_epoch}...")
    
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        epoch_loss = 0.0
        
        for i, (img, target) in enumerate(train_loader):
            img = img.to(device)
            target = target.to(device)
            
            output = model(img)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (i + 1) % 50 == 0:
                print(f"Époque [{epoch}/{end_epoch}] | Image [{i+1}/{len(train_loader)}] | Erreur (Loss): {loss.item():.6f}")
        
        avg_loss = epoch_loss / len(train_loader)
        save_path = f'models/csrnet_epoch_{epoch}.pth'
        torch.save(model.state_dict(), save_path)
        
        print(f"✅ Époque {epoch} terminée ! Erreur moyenne : {avg_loss:.6f} | Sauvegardé dans {save_path}\n")

if __name__ == '__main__':
    # Tu peux changer le 5 ici si tu veux faire tourner 10 ou 20 époques d'un coup !
    resume_training_auto(epochs_to_add=5)