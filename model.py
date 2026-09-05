import torch
import torch.nn as nn
from torchvision import models

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        
        # 'M' = MaxPooling (réduction de taille de l'image)
        # Nombres = Nombre de filtres (canaux)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        
        # 1. Le Front-end (Extracteur de caractéristiques)
        self.frontend = make_layers(self.frontend_feat)
        
        # 2. Le Back-end (Générateur de carte de densité avec Convolutions Dilatées)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        
        # 3. Couche de sortie (Fusionne les 64 canaux finaux en 1 seule image thermique)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        # Application des bonnes pratiques : Initialisation intelligente des poids
        self._initialize_weights()

    def forward(self, x):
        """
        C'est le chemin que prend l'image (x) à travers le réseau.
        """
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        """
        SOLUTION OPTIMALE : Au lieu de commencer avec des poids aléatoires,
        on charge les poids d'un VGG-16 pré-entraîné sur des millions d'images (ImageNet).
        Le modèle sait donc DÉJÀ reconnaître des formes (cercles, textures, têtes).
        """
        # Téléchargement des poids officiels de PyTorch
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # Injection des poids dans notre frontend
        frontend_state_dict = self.frontend.state_dict()
        vgg16_keys = list(vgg16.features.state_dict().keys())
        
        for i, key in enumerate(frontend_state_dict.keys()):
            frontend_state_dict[key] = vgg16.features.state_dict()[vgg16_keys[i]]
        self.frontend.load_state_dict(frontend_state_dict)
        
        # Le backend est nouveau, on l'initialise mathématiquement proprement (loi normale)
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.constant_(self.output_layer.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    """
    Fonction utilitaire pour assembler les blocs du réseau dynamiquement.
    """
    layers = []
    # C'est ici qu'intervient la magie des convolutions dilatées (d_rate = 2)
    d_rate = 2 if dilation else 1
        
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            
    return nn.Sequential(*layers)

# --- Bloc de test rapide ---
if __name__ == '__main__':
    # On simule la caméra qui envoie une image : (1 batch, 3 couleurs RGB, 768 hauteur, 1024 largeur)
    dummy_image = torch.randn(1, 3, 768, 1024)
    print("Création du modèle CSRNet en cours (téléchargement du VGG-16 si première fois)...")
    
    model = CSRNet()
    
    print("\nPassage de l'image dans le modèle (Inférence)...")
    output_map = model(dummy_image)
    
    print(f"Taille de l'image en entrée : {dummy_image.shape}")
    print(f"Taille de la carte thermique en sortie : {output_map.shape}")