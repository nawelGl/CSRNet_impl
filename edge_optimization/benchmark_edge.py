"""
benchmark_edge.py — Benchmark en environnement edge contraint (Docker + cgroups).

Script AUTONOME : aucune dépendance PyTorch.
Conçu pour tourner dans le conteneur Docker défini par le Dockerfile.

Fonctionnement :
  1. Détecte automatiquement les contraintes CPU/RAM imposées par Docker (cgroups)
  2. Aligne le nombre de threads ONNX Runtime sur le quota CPU
  3. Compare ONNX FP32 vs ONNX INT8 (statique) sur les images test Part B
  4. Mesure : latence (ms), MAE, RMSE, mémoire pic (MB), FPS
  5. Exporte les résultats en CSV

Usage (depuis le conteneur Docker) :
    python benchmark_edge.py
    python benchmark_edge.py --threads 2    # Forcer 2 threads
"""

import os
import sys
import time
import math
import csv
import resource
import argparse
import glob
import cv2
import h5py
import numpy as np
import onnxruntime as ort


# ============================================================================
# Détection des contraintes cgroups (Docker)
# ============================================================================

def detect_cgroup_constraints():
    """
    Lit les fichiers cgroups v1 et v2 pour déterminer les limites
    CPU et RAM imposées par Docker (--cpus et --memory).
    """
    cpu_limit = os.cpu_count()
    mem_limit_mb = None

    # --- CPU : cgroups v2 ---
    try:
        with open('/sys/fs/cgroup/cpu.max') as f:
            parts = f.read().strip().split()
            if parts[0] != 'max':
                cpu_limit = int(parts[0]) / int(parts[1])
    except (FileNotFoundError, PermissionError):
        pass

    # --- CPU : cgroups v1 (fallback) ---
    try:
        with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as f:
            quota = int(f.read().strip())
        with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as f:
            period = int(f.read().strip())
        if quota > 0:
            cpu_limit = quota / period
    except (FileNotFoundError, PermissionError):
        pass

    # --- RAM : cgroups v2 ---
    try:
        with open('/sys/fs/cgroup/memory.max') as f:
            val = f.read().strip()
            if val != 'max':
                mem_limit_mb = int(val) / (1024 * 1024)
    except (FileNotFoundError, PermissionError):
        pass

    # --- RAM : cgroups v1 (fallback) ---
    try:
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes') as f:
            val = int(f.read().strip())
            if val < 2**62:  # Valeur non-max = limite réelle
                mem_limit_mb = val / (1024 * 1024)
    except (FileNotFoundError, PermissionError):
        pass

    return cpu_limit, mem_limit_mb


def get_peak_memory_mb():
    """Retourne le pic de mémoire RSS du processus en MB (Linux)."""
    # Sur Linux, ru_maxrss retourne des KB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ============================================================================
# Prétraitement (numpy pur, pas de torchvision)
# ============================================================================

# Valeurs de normalisation ImageNet
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def preprocess_image(img_path, resize=None):
    """
    Charge et prétraite une image.
    Reproduit exactement le pipeline torchvision.ToTensor() + Normalize().
    Si resize=(W, H), redimensionne l'image avant le prétraitement.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize is not None:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0       # [0, 1]
    img = np.transpose(img, (2, 0, 1))          # HWC -> CHW
    img = np.expand_dims(img, axis=0)            # -> BCHW
    img = (img - MEAN) / STD                     # Normalisation ImageNet
    return img


def get_ground_truth(img_path):
    """Lit le .h5 et retourne le nombre réel de personnes."""
    h5_path = img_path.replace('.jpg', '.h5')
    with h5py.File(h5_path, 'r') as hf:
        return float(np.sum(hf['density'][:]))


# ============================================================================
# Benchmark ONNX
# ============================================================================

def benchmark_onnx(model_path, img_paths, label, num_threads, resize=None):
    """Évalue un modèle ONNX et retourne les métriques."""
    print(f"\n--- {label} ---")

    if not os.path.exists(model_path):
        print(f"  SKIP : {model_path} introuvable.")
        return None

    # Configuration alignée sur les contraintes cgroups
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = num_threads
    session_options.inter_op_num_threads = 1  # Un seul thread inter-op pour éviter la contention

    session = ort.InferenceSession(
        model_path, session_options,
        providers=['CPUExecutionProvider']
    )
    input_name = session.get_inputs()[0].name
    print(f"  Threads : intra={num_threads}, inter=1", flush=True)
    print(f"  Modèle chargé : {model_path} ({os.path.getsize(model_path)/(1024*1024):.1f} MB)", flush=True)

    # Warmup (3 images)
    print(f"  Warmup (3 images)...", flush=True)
    for j, path in enumerate(img_paths[:3]):
        img = preprocess_image(path, resize=resize)
        session.run(None, {input_name: img})
        print(f"    Warmup {j+1}/3 OK", flush=True)
    print(f"  Warmup terminé. Lancement du benchmark sur {len(img_paths)} images...", flush=True)

    # Benchmark
    mae_total = 0.0
    mse_total = 0.0
    latencies = []

    for i, path in enumerate(img_paths):
        img = preprocess_image(path, resize=resize)
        gt_count = get_ground_truth(path)

        start = time.perf_counter()
        output = session.run(None, {input_name: img})[0]
        elapsed = time.perf_counter() - start

        pred_count = float(np.sum(output))
        error = gt_count - pred_count

        mae_total += abs(error)
        mse_total += error ** 2
        latencies.append(elapsed * 1000)  # en ms

        if (i + 1) % 10 == 0 or (i + 1) == len(img_paths):
            avg_so_far = sum(latencies) / len(latencies)
            remaining = (len(img_paths) - (i + 1)) * avg_so_far / 1000
            print(f"  [{i+1}/{len(img_paths)}] {elapsed*1000:.0f} ms | "
                  f"moy: {avg_so_far:.0f} ms | reste: ~{remaining:.0f}s", flush=True)

    n = len(img_paths)
    latencies_sorted = sorted(latencies)

    peak_mem = get_peak_memory_mb()

    return {
        'name': label,
        'mae': mae_total / n,
        'rmse': math.sqrt(mse_total / n),
        'avg_ms': sum(latencies) / n,
        'p50_ms': latencies_sorted[int(n * 0.50)],
        'p95_ms': latencies_sorted[int(n * 0.95)],
        'fps': 1000.0 / (sum(latencies) / n),
        'peak_mem_mb': peak_mem,
    }


# ============================================================================
# Affichage et export
# ============================================================================

def print_results(results, cpu_limit, mem_limit_mb):
    """Affiche le tableau de résultats avec le contexte de contraintes."""
    print("\n" + "=" * 80)
    print("  RÉSULTATS DU BENCHMARK EDGE")
    print(f"  Contraintes : CPU = {cpu_limit:.1f} coeur(s)"
          f", RAM = {mem_limit_mb:.0f} MB" if mem_limit_mb else
          f"  Contraintes : CPU = {cpu_limit:.1f} coeur(s), RAM = illimitée")
    print("=" * 80)

    header = (f"  {'MODÈLE':<25} {'MAE':>6} {'RMSE':>6} "
              f"{'ms/img':>8} {'p95':>8} {'FPS':>6} {'RAM pic':>9}")
    print(header)
    print("-" * 80)

    for r in results:
        print(f"  {r['name']:<25} {r['mae']:>6.2f} {r['rmse']:>6.2f} "
              f"{r['avg_ms']:>8.1f} {r['p95_ms']:>8.1f} {r['fps']:>6.1f} "
              f"{r['peak_mem_mb']:>7.0f} MB")

    print("=" * 80)

    # Speedup INT8 vs FP32
    if len(results) == 2:
        speedup = results[0]['avg_ms'] / results[1]['avg_ms']
        mae_delta = results[1]['mae'] - results[0]['mae']
        print(f"\n  Speedup INT8 vs FP32 : x{speedup:.2f}")
        print(f"  Dégradation MAE     : +{mae_delta:.2f} personnes ({mae_delta/results[0]['mae']*100:.1f}%)")


def export_csv(results, output_path, cpu_limit, mem_limit_mb):
    """Exporte en CSV avec les contraintes en en-tête."""
    with open(output_path, 'w', newline='') as f:
        # Ligne de contexte
        f.write(f"# Contraintes: CPU={cpu_limit:.1f} cores")
        if mem_limit_mb:
            f.write(f", RAM={mem_limit_mb:.0f}MB")
        f.write("\n")

        writer = csv.DictWriter(f, fieldnames=[
            'name', 'mae', 'rmse', 'avg_ms', 'p50_ms', 'p95_ms', 'fps', 'peak_mem_mb'
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: f"{v:.2f}" if isinstance(v, float) else v for k, v in r.items()})

    print(f"\nCSV exporté : {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark Edge CSRNet (ONNX)")
    parser.add_argument('--threads', type=int, default=None,
                        help="Nombre de threads intra-op (défaut: auto depuis cgroups)")
    parser.add_argument('--dataset', type=str,
                        default='../data/ShanghaiTech_Dataset',
                        help="Chemin vers le dataset")
    parser.add_argument('--resize', type=int, nargs=2, default=None,
                        metavar=('W', 'H'),
                        help="Redimensionner les images (ex: --resize 640 480)")
    args = parser.parse_args()

    # --- Détection des contraintes ---
    cpu_limit, mem_limit_mb = detect_cgroup_constraints()
    num_threads = args.threads or max(1, int(cpu_limit))
    resize = tuple(args.resize) if args.resize else None

    print("=" * 50)
    print("  BENCHMARK EDGE — CSRNet ONNX")
    print("=" * 50)
    print(f"  CPU détecté     : {cpu_limit:.1f} coeur(s)")
    if mem_limit_mb:
        print(f"  RAM détectée    : {mem_limit_mb:.0f} MB")
    else:
        print(f"  RAM détectée    : illimitée")
    print(f"  Threads ONNX    : {num_threads}")
    print(f"  Résolution      : {f'{resize[0]}x{resize[1]}' if resize else 'native'}")
    print(f"  ONNX Runtime    : {ort.__version__}")

    # --- Chargement des images test ---
    test_dir = os.path.join(args.dataset, 'part_B_final', 'test_data', 'images')
    img_paths = sorted([
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir) if f.endswith('.jpg')
    ])
    print(f"  Images test     : {len(img_paths)}")

    if len(img_paths) == 0:
        print(f"\nErreur : aucune image dans {test_dir}")
        return

    # --- Benchmarks ---
    results = []

    res_tag = f" [{resize[0]}x{resize[1]}]" if resize else ""

    r1 = benchmark_onnx("csrnet.onnx", img_paths, f"ONNX FP32{res_tag}", num_threads, resize=resize)
    if r1:
        results.append(r1)

    r2 = benchmark_onnx("csrnet_quantized_static.onnx", img_paths, f"ONNX INT8{res_tag}", num_threads, resize=resize)
    if r2:
        results.append(r2)

    if not results:
        print("Aucun modèle ONNX trouvé. Vérifie que les fichiers .onnx sont dans le répertoire courant.")
        return

    # --- Résultats ---
    print_results(results, cpu_limit, mem_limit_mb)

    # Nom du CSV basé sur les contraintes et la résolution
    tag = f"cpu{cpu_limit:.0f}"
    if mem_limit_mb:
        tag += f"_ram{mem_limit_mb:.0f}"
    if resize:
        tag += f"_{resize[0]}x{resize[1]}"
    export_csv(results, f"benchmark_edge_{tag}.csv", cpu_limit, mem_limit_mb)


if __name__ == '__main__':
    main()