import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CSRNet
from dataset import ShanghaiTechDataset

def train():
    # --- 1. CONFIGURATION MATÉRIELLE (La Bonne Pratique) ---
    # PyTorch peut utiliser la carte graphique pour calculer 100x plus vite.
    # Sur ton Mac, on utilise l'accélération 'mps' (Metal Performance Shaders) si dispo, sinon le CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Accélération matérielle Apple Silicon (MPS) activée !")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Entraînement sur CPU.")

    # --- 2. PRÉPARATION DES DONNÉES ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(current_dir, 'data', 'ShanghaiTech_Dataset')
    
    # On charge uniquement les données d'entraînement (Train split)
    train_dataset = ShanghaiTechDataset(root_path=dataset_root, part='part_A_final', split='train_data')
    
    # Le DataLoader s'occupe de piocher les images. 
    # ASTUCE : batch_size=1 est OBLIGATOIRE ici car les images de ShanghaiTech 
    # n'ont pas toutes la même résolution (largeur/hauteur).
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # --- 3. INITIALISATION DU MODÈLE ET MATHÉMATIQUES ---
    model = CSRNet().to(device) # On envoie le modèle sur la puce graphique
    
    # La fonction d'erreur (Loss) : MSE (Mean Squared Error). 
    # Elle compare pixel par pixel la carte thermique de l'IA et notre corrigé .h5.
    criterion = nn.MSELoss()
    
    # L'Optimiseur : C'est l'algorithme qui modifie les poids du modèle (Backpropagation).
    # Le learning_rate (lr) est la "vitesse" d'apprentissage. S'il est trop grand, l'IA devient folle.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Création du dossier de sauvegarde pour les "cerveaux" entraînés
    os.makedirs('models', exist_ok=True)

    # --- 4. LA BOUCLE D'APPRENTISSAGE ---
    num_epochs = 5 # Nombre de fois où l'IA va voir le dataset complet
    
    print(f"\n🧠 Début de l'entraînement pour {num_epochs} époques...")
    
    for epoch in range(num_epochs):
        model.train() # Mode entraînement activé
        epoch_loss = 0.0
        
        for i, (img, target) in enumerate(train_loader):
            # On envoie l'image et le corrigé sur la carte graphique
            img = img.to(device)
            target = target.to(device)
            
            # Étape A : Prédiction (Forward pass)
            output = model(img)
            
            # Étape B : Calcul de l'erreur (Loss)
            loss = criterion(output, target)
            
            # Étape C : Ajustement des neurones (Backward pass)
            optimizer.zero_grad() # On nettoie l'ancienne erreur
            loss.backward()       # On calcule dans quel sens modifier les poids
            optimizer.step()      # On applique la modification !
            
            epoch_loss += loss.item()
            
            # Affichage de la progression toutes les 50 images
            if (i + 1) % 50 == 0:
                print(f"Époque [{epoch+1}/{num_epochs}] | Image [{i+1}/{len(train_loader)}] | Erreur (Loss): {loss.item():.6f}")
        
        # --- 5. SAUVEGARDE DU MODÈLE ---
        # À la fin de chaque époque, l'IA est un peu plus intelligente. On sauvegarde son cerveau !
        avg_loss = epoch_loss / len(train_loader)
        save_path = f'models/csrnet_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), save_path)
        
        print(f"✅ Époque {epoch+1} terminée ! Erreur moyenne : {avg_loss:.6f} | Sauvegardé dans {save_path}\n")

if __name__ == '__main__':
    train()