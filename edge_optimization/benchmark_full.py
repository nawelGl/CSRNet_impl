"""
benchmark_full.py — Benchmark comparatif complet pour le mémoire.

Compare 3 variantes du modèle sur les 316 images test de la Part B :
  1. PyTorch FP32 (CPU)  — la baseline
  2. ONNX FP32 (CPU)     — optimisations de graphe uniquement
  3. ONNX INT8 statique   — quantification des convolutions + activations

Pour chaque variante, mesure :
  - MAE  (Mean Absolute Error)  : précision moyenne du comptage
  - RMSE (Root Mean Squared Error) : sensibilité aux grosses erreurs
  - Latence moyenne (ms/image)
  - Débit (FPS)

Les résultats sont affichés en tableau et exportés en CSV pour le mémoire.

IMPORTANT : tout tourne sur CPU pour une comparaison équitable,
puisque la cible edge n'a pas de GPU.

Usage :
    $ python benchmark_full.py
"""

import os
import sys
import time
import math
import csv
import cv2
import h5py
import numpy as np
import torch
from torchvision import transforms
import onnxruntime as ort

# Ajout du chemin parent pour importer CSRNet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import CSRNet


# ============================================================================
# Prétraitement commun (identique pour PyTorch et ONNX)
# ============================================================================

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_test_data(dataset_path, part='part_B_final'):
    """Charge les chemins des images test et leurs ground truth."""
    test_dir = os.path.join(dataset_path, part, 'test_data', 'images')
    img_paths = sorted([
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir) if f.endswith('.jpg')
    ])
    print(f"Images de test chargées : {len(img_paths)} ({part})")
    return img_paths


def get_ground_truth(img_path):
    """Lit le fichier .h5 et retourne le nombre réel de personnes."""
    h5_path = img_path.replace('.jpg', '.h5')
    with h5py.File(h5_path, 'r') as hf:
        return float(np.sum(hf['density'][:]))


def preprocess_image(img_path):
    """Charge et prétraite une image (retourne un tenseur PyTorch)."""
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    return TRANSFORM(img).unsqueeze(0)  # [1, 3, H, W]


# ============================================================================
# Benchmark PyTorch
# ============================================================================

def benchmark_pytorch(model_path, img_paths):
    """Évalue le modèle PyTorch FP32 sur CPU."""
    print("\n--- PyTorch FP32 (CPU) ---")

    model = CSRNet().cpu()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Warmup (3 images pour stabiliser les caches CPU)
    with torch.no_grad():
        for path in img_paths[:3]:
            model(preprocess_image(path))

    mae_total = 0.0
    mse_total = 0.0
    time_total = 0.0

    with torch.no_grad():
        for i, path in enumerate(img_paths):
            tensor = preprocess_image(path)
            gt_count = get_ground_truth(path)

            start = time.perf_counter()
            output = model(tensor)
            elapsed = time.perf_counter() - start

            pred_count = output.sum().item()
            error = gt_count - pred_count

            mae_total += abs(error)
            mse_total += error ** 2
            time_total += elapsed

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(img_paths)}]")

    n = len(img_paths)
    return {
        'name': 'PyTorch FP32 (CPU)',
        'mae': mae_total / n,
        'rmse': math.sqrt(mse_total / n),
        'avg_ms': (time_total / n) * 1000,
        'fps': n / time_total,
    }


# ============================================================================
# Benchmark ONNX (FP32 ou INT8)
# ============================================================================

def benchmark_onnx(model_path, img_paths, label):
    """Évalue un modèle ONNX sur CPU."""
    print(f"\n--- {label} ---")

    if not os.path.exists(model_path):
        print(f"  SKIP : {model_path} introuvable.")
        return None

    # Configuration ONNX Runtime
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = os.cpu_count()  # Exploiter tous les coeurs

    session = ort.InferenceSession(
        model_path, session_options,
        providers=['CPUExecutionProvider']
    )

    input_name = session.get_inputs()[0].name

    # Warmup
    for path in img_paths[:3]:
        tensor_np = preprocess_image(path).numpy()
        session.run(None, {input_name: tensor_np})

    mae_total = 0.0
    mse_total = 0.0
    time_total = 0.0

    for i, path in enumerate(img_paths):
        tensor_np = preprocess_image(path).numpy()
        gt_count = get_ground_truth(path)

        start = time.perf_counter()
        output = session.run(None, {input_name: tensor_np})[0]
        elapsed = time.perf_counter() - start

        pred_count = float(np.sum(output))
        error = gt_count - pred_count

        mae_total += abs(error)
        mse_total += error ** 2
        time_total += elapsed

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(img_paths)}]")

    n = len(img_paths)
    return {
        'name': label,
        'mae': mae_total / n,
        'rmse': math.sqrt(mse_total / n),
        'avg_ms': (time_total / n) * 1000,
        'fps': n / time_total,
    }


# ============================================================================
# Affichage et export des résultats
# ============================================================================

def print_results(results):
    """Affiche un tableau comparatif propre."""
    print("\n" + "=" * 72)
    print(f"  {'MODÈLE':<28} {'MAE':>8} {'RMSE':>8} {'ms/img':>10} {'FPS':>8}")
    print("-" * 72)
    for r in results:
        print(f"  {r['name']:<28} {r['mae']:>8.2f} {r['rmse']:>8.2f} {r['avg_ms']:>10.1f} {r['fps']:>8.1f}")
    print("=" * 72)

    # Speedup relatif
    if len(results) >= 2:
        baseline_ms = results[0]['avg_ms']
        print("\n  Speedup par rapport à la baseline PyTorch :")
        for r in results[1:]:
            speedup = baseline_ms / r['avg_ms']
            print(f"    {r['name']:<28} : x{speedup:.2f}")


def export_csv(results, output_path):
    """Exporte les résultats en CSV pour inclusion dans le mémoire."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'mae', 'rmse', 'avg_ms', 'fps'])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'name': r['name'],
                'mae': f"{r['mae']:.2f}",
                'rmse': f"{r['rmse']:.2f}",
                'avg_ms': f"{r['avg_ms']:.1f}",
                'fps': f"{r['fps']:.1f}",
            })
    print(f"\nRésultats exportés dans : {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '..', 'data', 'ShanghaiTech_Dataset')

    # Chemins des modèles
    pytorch_model = os.path.join(script_dir, '..', 'models', 'csrnet_best.pth')
    onnx_fp32     = os.path.join(script_dir, 'csrnet.onnx')
    onnx_int8     = os.path.join(script_dir, 'csrnet_quantized_static.onnx')

    # Chargement des données de test
    img_paths = load_test_data(dataset_path, part='part_B_final')

    if len(img_paths) == 0:
        print("Aucune image test trouvée. Vérifie le chemin du dataset.")
        return

    # --- Exécution des 3 benchmarks ---
    results = []

    r1 = benchmark_pytorch(pytorch_model, img_paths)
    results.append(r1)

    r2 = benchmark_onnx(onnx_fp32, img_paths, "ONNX FP32")
    if r2:
        results.append(r2)

    r3 = benchmark_onnx(onnx_int8, img_paths, "ONNX INT8 (Static)")
    if r3:
        results.append(r3)

    # --- Résultats ---
    print_results(results)
    export_csv(results, os.path.join(script_dir, 'benchmark_results.csv'))


if __name__ == '__main__':
    main()