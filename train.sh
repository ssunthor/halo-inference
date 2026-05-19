#!/bin/bash
set -e

# Set this to the number of physical cores you want to donate (just in case it falls back to CPU)
CORES=8

echo "----------------------------------------------------------------"
echo "Starting Training Container (GPU Enabled)"
echo "Targeting Scikit-Learn 1.3.2 for Morpheus Compatibility"
echo "----------------------------------------------------------------"

# Changes made:
# 1. Added '-c rapidsai': This is the official channel for cuml
# 2. Added 'cuml': Installs the GPU algorithms
# 3. REMOVED 'morpheus-dfp': Not needed for extract.py, and was causing the version conflict
# 4. PINNED 'scikit-learn=1.3.2': Ensures models load correctly in the inference container

docker run -it --gpus all \
  -v "$(pwd)":/app \
  -w /app \
  --cpus="$CORES" \
  -e OMP_NUM_THREADS="$CORES" \
  nvcr.io/nvidia/morpheus/morpheus:25.06-runtime \
  bash -c "conda install -y -c rapidsai -c conda-forge -c nvidia \
  cuml scikit-learn=1.3.2 skl2onnx onnx requests mlflow onnxconverter-common && \
  python extract.py $*"