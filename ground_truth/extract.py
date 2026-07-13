import os; import yaml; import argparse

from kipo.utils import load_data, drop_cols, encode_target, preprocessing
from utils import compute_ground_truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type = str, required = True, help = "name of the dataset file")
    args = parser.parse_args()

    DATASET = args.file if args.file.endswith(".csv") else args.file + ".csv"

    with open(os.path.join(os.path.dirname(__file__), "..", "datasets", "config", f"{os.path.splitext(DATASET)[0]}.yml")) as f:
        conf = yaml.safe_load(f)

    cat_cols = conf.get("cat_cols", []); num_cols = conf.get("num_cols", [])

    data = load_data(os.path.join(os.path.dirname(__file__), "..", "datasets", "data", DATASET), conf)
    X, y = drop_cols(data, conf)

    X_scaled, _ = preprocessing(X, conf)
    y_enc = encode_target(y, conf)

    ground_truth = compute_ground_truth(X_scaled, y_enc)
    print(f"[INFO] ground truth rankings computed for dataset: {DATASET}")

    print(ground_truth)

    os.makedirs("data", exist_ok = True); path = os.path.join("ground_truth", DATASET)
    ground_truth.to_csv(path, index = False)
    print(f"[INFO] ground truth rankings saved at: {path}")


if __name__ == "main":
    main()