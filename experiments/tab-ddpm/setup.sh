#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

python -m pip install --upgrade pip

pip install \
    catboost==1.2.8 \
    category-encoders==2.3.0 \
    dython==0.5.1 \
    icecream==2.1.2 \
    libzero==0.0.8 \
    optuna==2.10.1 \
    pandas==1.3.5 \
    pyarrow==6.0.0 \
    rtdl==0.0.9 \
    scikit-learn==1.1.3 \
    skorch==0.11.0 \
    tomli-w==0.4.0 \
    tomli==1.2.2 \
    tqdm==4.62.3 \
    imbalanced-learn==0.11.0


check_directory() {
    [[ -d "$1" ]] && [[ "$(find "$1" -maxdepth 1 -type f | wc -l)" -eq "$2" ]]
}

DATASET_ROOT="../../datasets"

DATA_DIR="${DATASET_ROOT}/data"
CONFIG_DIR="${DATASET_ROOT}/config"

if ! check_directory "$DATA_DIR" 20 || ! check_directory "$CONFIG_DIR" 20; then
    echo "[ERROR] Required dataset artifacts are missing. Install the raw evaluated datasets before installing TabDDPM configuration files. See the 'Datasets' section of the README."
    exit 1
fi

LOCAL_DATASET_ROOT="./datasets"
LOCAL_DATA_DIR="${LOCAL_DATASET_ROOT}/data"
LOCAL_CONFIG_DIR="${LOCAL_DATASET_ROOT}/config"

DROPBOX_URL="https://www.dropbox.com/scl/fi/vl6mc3tlfcn37pts9pcd5/tabddpm_conf.zip?rlkey=uz6hhucnt8cp9nu2gck7d6urm&st=7r3ga28i&dl=0"

ZIP_FILE="/tmp/tabddpm_config.zip"
TMP_DIR="/tmp/tabddpm_extract"


mkdir -p "$LOCAL_DATA_DIR" "$LOCAL_CONFIG_DIR"

wget -q "$DROPBOX_URL" -O "$ZIP_FILE"

rm -rf "$TMP_DIR"
unzip -q "$ZIP_FILE" -d "$TMP_DIR"


find "$TMP_DIR" -name "*.toml" -exec cp {} "$LOCAL_CONFIG_DIR" \;

INFO_DIR="$TMP_DIR/tabddpm_conf/info"

if [[ -d "$INFO_DIR" ]]; then
    for json_file in "$INFO_DIR"/*/*.json; do
        [[ -f "$json_file" ]] || continue

        dataset_name=$(basename "$(dirname "$json_file")")
        target_dir="${LOCAL_DATA_DIR}/${dataset_name}"

        mkdir -p "$target_dir"

        cp "$json_file" "$target_dir/"
    done
fi


rm -rf "$ZIP_FILE" "$TMP_DIR"i


echo "[INFO] TabDDPM setup completed successfully."