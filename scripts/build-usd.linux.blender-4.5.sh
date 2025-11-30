#! /usr/bin/env bash
set -euo pipefail

# settings
VENV_DIR=".venv"
USD_REPO_DIR="external/OpenUSD"
BUILD_DIR="build/OpenUSD"
INSTALL_DIR="dependencies/OpenUSD"

MAX_JOBS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -n "$MAX_JOBS" ]; then
    JOBS=$MAX_JOBS
elif command -v nproc &> /dev/null; then
    JOBS=$(nproc)
elif command -v sysctl &> /dev/null; then
    JOBS=$(sysctl -n hw.ncpu)
else
    JOBS=4
fi

# create venv
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment..."
    python -m venv "$VENV_DIR"
else
    echo "[INFO] Virtual environment already exists. Skipping creation."
fi

# activate venv
source "$VENV_DIR/bin/activate"

# install dependencies
echo "[INFO] Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -U PyOpenGL PySide6

# prepare python info
PYTHON_EXEC=$(which python)
PYTHON_VERSION=$($PYTHON_EXEC -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
PYTHON_INCLUDE_DIR=$($PYTHON_EXEC -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIBRARY=$($PYTHON_EXEC -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') + '/' + sysconfig.get_config_var('LDLIBRARY'))")

echo "[INFO] Using Python from venv:"
echo "  Executable: $PYTHON_EXEC"
echo "  Include:    $PYTHON_INCLUDE_DIR"
echo "  Library:    $PYTHON_LIBRARY"
echo

# checkout USD
cd "$USD_REPO_DIR"
git checkout .
git checkout "v25.02"
git apply --ignore-whitespace --whitespace=nowarn ../../scripts/build-usd.blender-4.5.patch
cd -

# build OpenUSD
echo "[INFO] Building OpenUSD..."

PXR_INTERNAL_NAMESPACE="pxrBlender_v25_02"

python "$USD_REPO_DIR/build_scripts/build_usd.py" \
    --build "$BUILD_DIR" \
    --build-monolithic \
    --usd-imaging \
    --onetbb \
    --no-python \
    --no-openvdb \
    --no-materialx \
    --no-examples \
    -j${JOBS} \
    --build-args \
        "USD,-DPXR_SET_INTERNAL_NAMESPACE=$PXR_INTERNAL_NAMESPACE" \
    --build-python-info \
        $PYTHON_EXEC \
        $PYTHON_INCLUDE_DIR \
        $PYTHON_LIBRARY \
        $PYTHON_VERSION \
    "$INSTALL_DIR"

echo "[INFO] Build completed successfully."
