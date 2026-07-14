#!/bin/bash

set -e

cd "$(dirname "$0")"

wget -qO datasets.zip "https://www.dropbox.com/scl/fi/mihlldhisdbxvaqvghrgi/datasets.zip?rlkey=3ban8ao5ag54fmsdnr2h256bn&st=dxtwgynf&dl=1"

unzip -qo datasets.zip -d temp
cp -r temp/datasets/* .
rm -rf temp datasets.zip

echo "[INFO] Dataset downloaded and extracted successfully."