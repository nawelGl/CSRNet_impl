import os
import glob
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CSRNet
from dataset import ShanghaiTechDataset

def resume_training_optimized(epochs_to_add=30):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Accélération matérielle : {device}")

    # préparation des données
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(current_dir, 'data', 'ShanghaiTech_Dataset')
    train_dataset = ShanghaiTechDataset(root_path=dataset_root, part='part_B_final', split='train_data')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = CSRNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # trouver la meilleure version du modèle
    fichiers_modeles = glob.glob('models/csrnet_epoch_*.pth')
    if not fichiers_modeles:
        print(" Aucun modèle d'historique trouvé. Lance d'abord train.py.")
        return
        
    dernier_modele = max(fichiers_modeles, key=lambda f: int(re.search(r'epoch_(\d+)', f).group(1)))
    start_epoch = int(re.search(r'epoch_(\d+)', dernier_modele).group(1)) + 1
    end_epoch = start_epoch + epochs_to_add - 1

    chemin_best = 'models/csrnet_best.pth'
    if os.path.exists(chemin_best):
        print(f" Fichier 'best' trouvé : époque {start_epoch}).")
        model.load_state_dict(torch.load(chemin_best, map_location=device))
    else:
        print(f" Pas de 'best' trouvé, reprise classique depuis : {dernier_modele}")
        model.load_state_dict(torch.load(dernier_modele, map_location=device))

    # enregistrement du meilleur modèle jusqu'ici
    chemin_record = 'models/best_loss.txt'
    if os.path.exists(chemin_record):
        with open(chemin_record, 'r') as f:
            meilleure_loss = float(f.read().strip())
        print(f" Record actuel à battre : {meilleure_loss:.6f}")
    else:
        meilleure_loss = float('inf')
        print(" Aucun record précédent enregistré. La prochaine époque sera le record de base.")

    print(f"\n Entraînement Optimisé (Époques {start_epoch} à {end_epoch})...")
    
    # boucle d'entrainement
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        epoch_loss = 0.0
        
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            output = model(img)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if (i + 1) % 50 == 0:
                print(f"Époque [{epoch}/{end_epoch}] | Image [{i+1}/{len(train_loader)}] | Erreur: {loss.item():.6f}")
        
        avg_loss = epoch_loss / len(train_loader)
        
        # ajustement de la vitesse si stagnation
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 1e-5:
            print(f" Vitesse d'apprentissage réduite à : {current_lr}")
        
        # sauvegarde classique
        save_path = f'models/csrnet_epoch_{epoch}.pth'
        torch.save(model.state_dict(), save_path)
        
        # check si meilleur modèle ou pas
        if avg_loss < meilleure_loss:
            meilleure_loss = avg_loss
            # sauvegarde si meilleur modèle
            torch.save(model.state_dict(), chemin_best)
            # Mise à jour du fichier texte du record
            with open(chemin_record, 'w') as f:
                f.write(str(meilleure_loss))
            tag_best = "  NOUVEAU RECORD  "
        else:
            tag_best = ""
            
        print(f" Époque {epoch} terminée, moyenne : {avg_loss:.6f}{tag_best}\n")

if __name__ == '__main__':
    # nombre d'époques qu'on ajoute en un lancement
    resume_training_optimized(epochs_to_add=2)