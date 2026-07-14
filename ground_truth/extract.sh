#!/bin/bash

cd "$(dirname "$0")"

datasets=(
    abalone
    car-eval-4
    ai4i2020
    airlines
    bank32nh
    bank-customer-churn-prediction
    bank-marketing
    churn
    fried
    japanese-vowels
    lines-overload-50
    magic-gamma-telescope
    mammography
    nhanes
    online-shoppers-purchasing-intention
    pen-digits
    rl
    seismic-bumps
    ur3-cobot-ops
    wilt
)

for dataset in "${datasets[@]}"
do
    echo "[INFO] Extracting ground truth for ${dataset}"
    python3 extract.py --file "${dataset}"
done

echo "[INFO] All datasets completed."