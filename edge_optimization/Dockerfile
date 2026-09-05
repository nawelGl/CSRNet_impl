# =============================================================================
# Dockerfile — Environnement Edge simulé pour CSRNet
#
# Image volontairement minimaliste : PAS de PyTorch, PAS de GPU.
# Seul ONNX Runtime est installé, comme sur un vrai dispositif edge
# (Raspberry Pi, passerelle industrielle, etc.)
#
# Build :
#   docker build -t csrnet-edge .
#
# Run (exemples avec contraintes cgroups) :
#
#   Profil "Edge Server" (4 coeurs, 4 Go RAM) :
#   docker run --rm --cpus="4.0" --memory="4g" \
#     -v "$(pwd)/..:/workspace" \
#     -w /workspace/edge_optimization \
#     csrnet-edge python benchmark_edge.py
#
#   Profil "Raspberry Pi 4" (2 coeurs, 2 Go RAM) :
#   docker run --rm --cpus="2.0" --memory="2g" \
#     -v "$(pwd)/..:/workspace" \
#     -w /workspace/edge_optimization \
#     csrnet-edge python benchmark_edge.py
#
#   Sans contrainte (toutes les ressources de l'hôte) :
#   docker run --rm \
#     -v "$(pwd)/..:/workspace" \
#     -w /workspace/edge_optimization \
#     csrnet-edge python benchmark_edge.py
# =============================================================================

FROM python:3.10-slim

# Dépendance système pour OpenCV (libglib nécessaire même en mode headless)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Dépendances Python — image légère (~200 MB vs ~2 GB avec PyTorch)
RUN pip install --no-cache-dir \
    onnxruntime \
    opencv-python-headless \
    numpy \
    h5py

WORKDIR /workspace