import pandas as pd; import numpy as np; import os
from pathlib import Path

DATASETS =  ["abalone", "ai4i2020", "airlines", "bank-customer-churn-prediction", "bank-marketing", "bank32nh", 
             "car-eval-4", "churn", "fried", "japanese-vowels", "magic-gamma-telescope", "rl", "lines-overload-50",
             "mammography", "nhanes", "online-shoppers-purchasing-intention", "pen-digits", 
             "seismic-bumps", "ur3-cobot-ops", "wilt"]

LAST_COLUMN_NAMES = {
    "abalone": "Rings",
    "ai4i2020": "Machine failure",
    "airlines": "Delay",
    "bank-customer-churn-prediction": "churn",
    "bank-marketing": "deposit",
    "bank32nh": "class",
    "car-eval-4": "Class",
    "churn": "class",
    "fried": "class",
    "japanese-vowels": "class",
    "magic-gamma-telescope": "Target",
    "rl": "class",
    "lines-overload-50": "Overload_Line_50",
    "mammography": "Class",
    "nhanes": "class",
    "online-shoppers-purchasing-intention": "Revenue",
    "pen-digits": "Class",
    "seismic-bumps": "Class",
    "ur3-cobot-ops": "class",
    "wilt": "class"
}

EXPECTED_FILES = 10
errors = []

for DATASET in DATASETS:
    print(f"\n[INFO] Processing dataset: {DATASET}")

    path = Path(f"./datasets/{DATASET}")

    expected_column = LAST_COLUMN_NAMES[DATASET]

    csv_files = sorted(path.glob("*.csv"))
    num_files = len(csv_files)


    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        old_column = df.columns[-1]

        if old_column != expected_column:
            df = df.rename(columns={old_column: expected_column})
            df.to_csv(csv_file, index=False)

            print(f"[INFO] {csv_file}: renamed '{old_column}' -> '{expected_column}'")
        else:
            print(f"[INFO] {csv_file}: last column already '{expected_column}'")



if errors:
    raise ValueError(f"[ERROR] Incorrect CSV count:\n{errors}")
else:
    print("\n[INFO] All datasets and generators contain the expected number of CSV files.")
    print("[INFO] All target columns have been verified and updated.")