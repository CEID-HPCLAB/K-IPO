import yaml; import argparse; import os

import numpy as np; import pandas as pd

from sklearn.model_selection import train_test_split

from kipo.utils import load_data, encode_target, extract_target_var


def load_config(path):
    with open(os.path.join(os.path.dirname(__file__), path)) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type = str, help = "Dataset name")
    args = parser.parse_args()
        
    DATASET = args.dataset

    run_config = load_config(os.path.join("..", "config.yml"))
    data_config = load_config(os.path.join("..", "..", "..", "datasets", "config", f"{DATASET}.yml"))
    
    SEED = run_config["seed"]; TRAIN_TEST_RATIO = run_config["train_test_ratio"]

    data = load_data(os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets", "data", DATASET), data_config)
    X, y = extract_target_var(data, data_config)
    
    y = encode_target(y, data_config)

    num_cols = data_config.get("num_cols", [])

    cat_cols_config = data_config.get("cat_cols", [])
    cat_cols = [list(cat.keys())[0] for cat in cat_cols_config] if cat_cols_config else []

    missing_num = set(num_cols) - set(X.columns); missing_cat = set(cat_cols) - set(X.columns)

    if missing_num:
        raise ValueError(f"Missing numerical columns: {missing_num}")

    if missing_cat:
        raise ValueError(f"Missing categorical columns: {missing_cat}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = TRAIN_TEST_RATIO, random_state = SEED, stratify = y)

    if num_cols:
        X_train_num = X_train[num_cols].values
        X_test_num = X_test[num_cols].values
    
    else:
        X_train_num = None; X_test_num = None

    if cat_cols:
        X_train_cat = X_train[cat_cols].values
        X_test_cat = X_test[cat_cols].values
    
    else:
        X_train_cat = None; X_test_cat = None
    
    y_train = y_train.values.reshape(-1); y_test = y_test.values.reshape(-1)

    OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "data", DATASET); os.makedirs(OUT_DIR, exist_ok = True)
    
    if X_train_num is not None:
        np.save(os.path.join(OUT_DIR, "X_num_train.npy"), X_train_num)
        np.save(os.path.join(OUT_DIR, "X_num_test.npy"), X_test_num)

    if X_train_cat is not None:
        np.save(os.path.join(OUT_DIR, "X_cat_train.npy"), X_train_cat)
        np.save(os.path.join(OUT_DIR, "X_cat_test.npy"), X_test_cat)
    
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)


if __name__ == '__main__':
    main()