import os
import glob
import re
import torch
import cv2
import numpy as np
from torchvision import transforms
from model import CSRNet

def run_video_inference():
    #TODO : modifier pour prendre le meilleur modèle et non le dernier
    # Sur M1, on utilise MPS pour le GPU, sinon CPU.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Inférence vidéo lancée sur : {device}")

    # recherche auto du meilleur modèle
    model = CSRNet().to(device)
    best_models = glob.glob('models/*best*.pth')
    
    if best_models:
        meilleur_modele = best_models[0]
        print(f"Meilleur modèle trouvé : {meilleur_modele}")
    else:
        # prendre le fichier le plus récent si aucun tag 'best' n'existe
        fichiers_modeles = glob.glob('models/*.pth')
        if not fichiers_modeles:
            print("Aucun modèle trouvé dans le dossier 'models/'. Impossible de lancer la vidéo.")
            return
            
        meilleur_modele = max(fichiers_modeles, key=os.path.getctime)
        print(f"Modèle 'best' introuvable. Chargement du plus récent : {meilleur_modele}")

    # chargement
    model.load_state_dict(torch.load(meilleur_modele, map_location=device))
    model.eval() 
    print(f"Modèle chargé ({meilleur_modele}) et verrouillé en mode Évaluation.")

    # initialisation webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Impossible d'ouvrir la webcam.")
        return

    # définition de la résolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # mêmes normalisations que pendant l'entraînement
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n📹 Flux vidéo activé. Appuie sur 'q' pour quitter.")

    # boucle infinie sur la vidéo
    frame_count = 0
    frequence_ia = 10  # inférence toutes les 10 frames
    
    # garder en mémoire la dernière prédiction
    dernier_compte = 0
    derniere_heatmap = None

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            display_frame = frame.copy()

            # declanchement csrnet 1/10 images
            if frame_count % frequence_ia == 0:
                img_or = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_tensor = transform(img_or).unsqueeze(0).to(device)
                
                output = model(img_tensor)
                pred_map = output.squeeze().cpu().numpy()
                dernier_compte = np.sum(pred_map)

                # génération de la nouvelle carte de chaleur
                heatmap = cv2.normalize(pred_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                derniere_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # affichage fluide continu : si on a déjà calculé au moins une heatmap, on la superpose
            if derniere_heatmap is not None:
                heatmap_resized = cv2.resize(derniere_heatmap, (display_frame.shape[1], display_frame.shape[0]))
                combined_frame = cv2.addWeighted(display_frame, 0.6, heatmap_resized, 0.4, 0)
            else:
                combined_frame = display_frame # Première seconde de la vidéo sans heatmap

            # affichage du texte
            text = f"Compte IA : {dernier_compte:.0f} personnes"
            cv2.putText(combined_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # affichage de l'état de calcul
            etat = "IA Active" if frame_count % frequence_ia == 0 else "IA en pause"
            couleur_etat = (0, 0, 255) if frame_count % frequence_ia == 0 else (0, 255, 0)
            cv2.putText(combined_frame, etat, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur_etat, 2)

            cv2.imshow("CSRNet - Temps Réel (Webcam)", combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # nettoyage des ressources
    cap.release()
    cv2.destroyAllWindows()
    print("\nFlux vidéo arrêté proprement.")

if __name__ == '__main__':
    run_video_inference()