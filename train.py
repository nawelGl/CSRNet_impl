import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CSRNet
from dataset import ShanghaiTechDataset

def train():
    # on utilise l'accélération mps si dispo, sinon le CPU pour améliorer la performance
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(" Accélération mps activée")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Entraînement sur CPU.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(current_dir, 'data', 'ShanghaiTech_Dataset')
    
    # on ne charge que les données d'entraînement
    train_dataset = ShanghaiTechDataset(root_path=dataset_root, part='part_A_final', split='train_data')
    
    # DataLoader pioche les images 
    # batch_size=1 obligatoire ici car toutes les images du jeu de données n'ont pas la même résolution
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = CSRNet().to(device) # On envoie le modèle sur la puce graphique
    
    # MSE de l'article
    criterion = nn.MSELoss()
    
    # optimiseur qui modifie les poids du modèle
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # dossier de sauvegarde
    os.makedirs('models', exist_ok=True)

    # boucle d'apprentissage
    num_epochs = 5 # nombre d'époques d'entrainement au lancement du script
    
    print(f"\n Début de l'entraînement pour {num_epochs} époques...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for i, (img, target) in enumerate(train_loader):
            # on envoie l'image et le corrigé sur la carte graphique
            img = img.to(device)
            target = target.to(device)
            
            # prédiction
            output = model(img)
            
            # calcul de l'erreur
            loss = criterion(output, target)
            
            # ajustement
            optimizer.zero_grad() # on nettoie l'ancienne erreur
            loss.backward()       # on calcule dans quel sens modifier les poids
            optimizer.step()      # on applique la modification
            
            epoch_loss += loss.item()
            
            # affichage de la progression toutes les 50 images
            if (i + 1) % 50 == 0:
                print(f"Époque [{epoch+1}/{num_epochs}] | Image [{i+1}/{len(train_loader)}] | Erreur (Loss): {loss.item():.6f}")
        
        # sauvegarde du model enrainé
        avg_loss = epoch_loss / len(train_loader)
        save_path = f'models/csrnet_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), save_path)
        
        print(f" Époque {epoch+1} terminée, erreur moyenne : {avg_loss:.6f} | Sauvegardé dans {save_path}\n")

if __name__ == '__main__':
    train()