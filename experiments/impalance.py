import os; import yaml; import pandas as pd; import numpy as np

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from imblearn.datasets import make_imbalance

from kipo.utils import load_data, extract_target_var

def main():

    def load_config():
        with open(os.path.join(os.path.dirname(__file__), "config.yml")) as f:
            return yaml.safe_load(f)

    run_config = load_config()
    
    DATASET = run_config["dataset"]
    IMBALANCE_RATIO = run_config["imbalance_ratio"]

    config_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "config", f"{os.path.splitext(DATASET)[0]}.yml")
    with open(config_path) as f:
        conf = yaml.safe_load(f)
    
    cat_cols = [list(col.keys())[0] for col in conf.get("cat_cols", [])]
    
    data = load_data(os.path.join(os.path.dirname(__file__), "..", "datasets", "data", DATASET), conf)
    X, y = extract_target_var(data, conf)

    data_init = pd.concat([X, y], axis = 1)

    target_col = data.columns[-1]

    X_enc = X.copy()
    if cat_cols:
        X_enc[cat_cols] = X_enc[cat_cols]
        encoder = OrdinalEncoder()
        X_enc[cat_cols] = encoder.fit_transform(X_enc[cat_cols])

    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    unique, counts = np.unique(y, return_counts = True)
    
    majority_class = unique[np.argmax(counts)]; minority_class = unique[np.argmin(counts)]
    majority_cnt = counts[np.argmax(counts)]; minority_cnt = max(1, min(int(majority_cnt * IMBALANCE_RATIO), counts[np.argmin(counts)]))

    sampling_strategy = {majority_class: majority_cnt, minority_class: minority_cnt}

    X_imb, y_imb = make_imbalance(X_enc.values, y, sampling_strategy = sampling_strategy)

    data_imb = pd.DataFrame(X_imb, columns = X_enc.columns); data_imb[target_col] = y_imb

    print(f"[INFO] dataset: {DATASET}")
    
    print(f"[INFO] initial class distribution: {dict(zip(unique, counts))}")
    print(f"[INFO] initial dataset shape: {data_init.shape}")
    
    unique_imb, counts_imb = np.unique(y_imb, return_counts = True)
    print(f"[INFO] class distribution after imbalance (ratio = {IMBALANCE_RATIO}): {dict(zip(unique_imb, counts_imb))}")
    print(f"[INFO] dataset shape after imbalance: {data_imb.shape}")


if __name__ == "__main__":
    main()