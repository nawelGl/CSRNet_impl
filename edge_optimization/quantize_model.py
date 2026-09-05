"""
quantize_model_static.py — Quantification STATIQUE INT8 du modèle CSRNet ONNX.

Pourquoi statique et pas dynamique ?
- quantize_dynamic ne quantifie que les couches MatMul/LSTM/Attention.
- CSRNet est un pur CNN (que des Conv2d). La quantification dynamique
  n'a donc quasi aucun effet sur ce type d'architecture.
- quantize_static quantifie AUSSI les convolutions et les activations,
  en s'appuyant sur un dataset de calibration pour mesurer les bornes
  [min, max] des activations intermédiaires.

Pré-requis :
    - csrnet.onnx généré par export_onnx.py (avec axes dynamiques)
    - Le dataset ShanghaiTech Part B accessible dans ../data/

Usage :
    $ python quantize_model_static.py
"""

import os
import sys
import glob
import cv2
import numpy as np

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)

# quant_pre_process peut être à deux endroits selon la version d'onnxruntime
try:
    from onnxruntime.quantization import quant_pre_process
except ImportError:
    from onnxruntime.quantization.shape_inference import quant_pre_process


# ============================================================================
# CalibrationDataReader : fournit des images réelles au processus de calibration
# ============================================================================

class ShanghaiTechCalibrationReader(CalibrationDataReader):
    """
    Lit un sous-ensemble d'images ShanghaiTech et les fournit une par une
    au processus de calibration d'ONNX Runtime.

    La calibration mesure, pour chaque couche du réseau, la distribution
    des valeurs d'activation sur des données réelles. Ces statistiques
    servent ensuite à calculer les facteurs d'échelle (scale) et points
    zéro (zero_point) optimaux pour la conversion FP32 -> INT8.
    """

    def __init__(self, dataset_path, part='part_B_final', split='train_data', num_samples=100):
        self.img_paths = sorted(glob.glob(
            os.path.join(dataset_path, part, split, 'images', '*.jpg')
        ))[:num_samples]

        if len(self.img_paths) == 0:
            raise FileNotFoundError(
                f"Aucune image trouvée dans {os.path.join(dataset_path, part, split, 'images')}. "
                "Vérifie le chemin vers le dataset."
            )

        print(f"Calibration : {len(self.img_paths)} images chargées depuis {part}/{split}")
        self.index = 0

        # Valeurs de normalisation ImageNet (identiques à l'entraînement)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def get_next(self):
        """Retourne l'image suivante prétraitée, ou None quand toutes ont été lues."""
        if self.index >= len(self.img_paths):
            return None

        img_path = self.img_paths[self.index]
        self.index += 1

        # Prétraitement identique au pipeline d'entraînement :
        # 1. Lecture BGR -> RGB
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Conversion [0, 255] uint8 -> [0.0, 1.0] float32
        img = img.astype(np.float32) / 255.0

        # 3. HWC -> CHW (format attendu par le réseau)
        img = np.transpose(img, (2, 0, 1))

        # 4. Ajout de la dimension batch : CHW -> BCHW
        img = np.expand_dims(img, axis=0)

        # 5. Normalisation ImageNet
        img = (img - self.mean) / self.std

        if self.index % 20 == 0:
            print(f"  Calibration : {self.index}/{len(self.img_paths)} images traitées...")

        return {'input': img}

    def rewind(self):
        """Remet le lecteur au début (utilisé si ONNX Runtime fait plusieurs passes)."""
        self.index = 0


# ============================================================================
# Fonction principale
# ============================================================================

def apply_static_quantization():
    # --- Chemins ---
    model_fp32 = "csrnet.onnx"
    model_preprocessed = "csrnet_preprocessed.onnx"
    model_int8 = "csrnet_quantized_static.onnx"

    if not os.path.exists(model_fp32):
        print(f"Erreur : {model_fp32} introuvable. Lance d'abord export_onnx.py.")
        return

    # Chemin vers le dataset (structure attendue : ../data/ShanghaiTech_Dataset/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '..', 'data', 'ShanghaiTech_Dataset')

    # --- Étape 1 : Prétraitement du graphe ONNX ---
    # quant_pre_process ajoute les informations de forme (shape inference)
    # et applique des optimisations de graphe nécessaires à la quantification.
    print("Étape 1/3 : Prétraitement du graphe ONNX...")
    quant_pre_process(model_fp32, model_preprocessed)
    print("  Graphe prétraité avec succès.")

    # --- Étape 2 : Création du lecteur de calibration ---
    print("\nÉtape 2/3 : Calibration sur les images de la Part B...")
    calibration_reader = ShanghaiTechCalibrationReader(
        dataset_path=dataset_path,
        part='part_B_final',
        split='train_data',
        num_samples=100  # 100 images suffisent pour une calibration robuste
    )

    # --- Étape 3 : Quantification statique ---
    print("\nÉtape 3/3 : Quantification statique INT8...")
    quantize_static(
        model_input=model_preprocessed,
        model_output=model_int8,
        calibration_data_reader=calibration_reader,

        # Format QDQ : insère des noeuds QuantizeLinear / DequantizeLinear
        # autour de chaque opération. C'est le format recommandé et le plus
        # portable (fonctionne sur CPU, GPU, et accélérateurs spécialisés).
        quant_format=QuantFormat.QDQ,

        # Quantification par canal : chaque filtre de convolution a ses propres
        # facteurs d'échelle, ce qui préserve mieux la précision qu'une
        # quantification par tenseur (per_tensor).
        per_channel=True,

        # Poids en INT8 signé [-128, 127]
        weight_type=QuantType.QInt8,

        # Activations en UINT8 non signé [0, 255]
        # Cohérent car les ReLU garantissent des activations >= 0
        activation_type=QuantType.QUInt8,
    )

    # --- Bilan ---
    size_fp32 = os.path.getsize(model_fp32) / (1024 * 1024)
    size_int8 = os.path.getsize(model_int8) / (1024 * 1024)
    reduction = (1 - size_int8 / size_fp32) * 100

    print(f"\n{'='*45}")
    print(f"  QUANTIFICATION STATIQUE TERMINÉE")
    print(f"{'='*45}")
    print(f"  Modèle FP32     : {size_fp32:.2f} MB")
    print(f"  Modèle INT8     : {size_int8:.2f} MB")
    print(f"  Réduction       : {reduction:.1f}%")
    print(f"  Fichier généré  : {model_int8}")

    # Nettoyage du fichier intermédiaire
    if os.path.exists(model_preprocessed):
        os.remove(model_preprocessed)
        print(f"\n  (Fichier intermédiaire {model_preprocessed} supprimé)")


if __name__ == '__main__':
    apply_static_quantization()