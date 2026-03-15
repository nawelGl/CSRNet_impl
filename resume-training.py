import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CSRNet
from dataset import ShanghaiTechDataset

def resume_training():
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

    # --- 3. INITIALISATION ET REPRISE (Le cœur du script) ---
    model = CSRNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # C'est ici qu'on charge le cerveau de l'époque 5
    checkpoint_path = 'models/csrnet_epoch_5.pth'
    if os.path.exists(checkpoint_path):
        # On injecte les poids sauvegardés dans notre modèle vierge
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"🔄 Reprise réussie : Cerveau chargé depuis {checkpoint_path}")
    else:
        print(f"❌ Erreur : Fichier {checkpoint_path} introuvable. As-tu bien fini tes 5 premières époques ?")
        return

    # --- 4. LA BOUCLE D'APPRENTISSAGE (Époques 6 à 10) ---
    start_epoch = 6
    end_epoch = 10
    
    print(f"\n🧠 Reprise de l'entraînement pour les époques {start_epoch} à {end_epoch}...")
    
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
    resume_training()