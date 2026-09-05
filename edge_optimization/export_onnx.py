"""
export_onnx.py — Export du modèle CSRNet PyTorch vers le format ONNX.

    $ python export_onnx.py
"""

import torch
import os
import sys

# Import de l'exporteur legacy pour contourner Dynamo (PyTorch 2.x)
import torch.onnx.utils as onnx_utils

# Ajout du chemin parent pour importer l'architecture CSRNet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import CSRNet


def convert_to_onnx():
    device = torch.device("cpu")  # L'export ONNX se fait toujours sur CPU

    # --- 1. Chargement du meilleur modèle entraîné ---
    model = CSRNet().to(device)
    chemin_modele = '../models/csrnet_best.pth'

    print(f"Chargement des poids depuis : {chemin_modele}")
    model.load_state_dict(torch.load(chemin_modele, map_location=device))
    model.eval()

    # --- 2. Entrée factice (définit la forme de référence, pas une contrainte) ---
    dummy_input = torch.randn(1, 3, 768, 1024, device=device)
    output_onnx = "csrnet.onnx"

    # --- 3. Axes dynamiques : hauteur et largeur variables ---
    # Cela permet au modèle ONNX de fonctionner avec n'importe quelle résolution :
    # - 768x1024 pour le benchmark sur ShanghaiTech
    # - 640x480 pour la démo webcam en direct
    dynamic_axes = {
        'input':  {2: 'height', 3: 'width'},
        'output': {2: 'out_height', 3: 'out_width'}
    }

    # --- 4. Export ---
    print("Exportation vers ONNX (axes dynamiques activés)...")
    with torch.no_grad():
        onnx_utils.export(
            model,
            dummy_input,
            output_onnx,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )

    taille_mb = os.path.getsize(output_onnx) / (1024 * 1024)
    print(f"\nExport réussi : {output_onnx} ({taille_mb:.2f} MB)")
    print("Le modèle accepte désormais n'importe quelle résolution d'entrée.")


if __name__ == '__main__':
    convert_to_onnx()