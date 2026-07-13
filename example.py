import logging; import warnings
import os; import yaml; import pandas as pd

from sklearn.model_selection import train_test_split

from kipo.selector import KIPOSelector as KIPO
from kipo.importance import compute_importance, kendall_tau
from kipo.generator import generate_data_pool
from kipo.utils import load_data, preprocessing, encode_target, extract_target_var, construct_metadata

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("smote_variants").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

def main():

    def load_config():
        with open(os.path.join(os.path.dirname(__file__), "config.yml")) as f:
            return yaml.safe_load(f)

    run_config = load_config()
    
    DATASET = run_config["dataset"]
    GENERATOR = run_config["generator"]
    TAU_THRESHOLD = run_config["tau_threshold"]
    TOPK_ORDERING = run_config["topk_ordering"]
    TOPK_OVERLAP = run_config["topk_overlap"]
    BALANCE_RATIO = run_config["balance_ratio"]
    TRAIN_TEST_RATIO = run_config["train_test_ratio"]
    SEED_GEN = run_config["seed"]
    WITHOUT_KIPO = run_config["without_kipo"]

    with open(os.path.join(os.path.dirname(__file__), "datasets", "config", f"{os.path.splitext(DATASET)[0]}.yml")) as f:
        conf = yaml.safe_load(f)

    cat_cols = [list(col.keys())[0] for col in conf.get("cat_cols", [])]; num_cols = conf.get("num_cols", [])

    if not bool(cat_cols) and "SMOTENC" == GENERATOR:
        raise ValueError("SMOTENC can't support datasets with only numerical features")
    
    if bool(num_cols) and "SMOTEN" == GENERATOR:
        raise ValueError("SMOTEN can't support datasets with numerical features") 
    
    if bool(cat_cols) and "SMOTEENN" == GENERATOR:
        raise ValueError("SMOTEENN can't support datasets with categorical features")
    
    if bool(cat_cols) and "SMOTEWB" == GENERATOR:
        raise ValueError("SMOTEWB can't support datasets with categorical features")
    

    data = load_data(os.path.join(os.path.dirname(__file__), "datasets", "data", DATASET), conf)
    X, y = extract_target_var(data, conf)
    
    y = encode_target(y, conf)

    majority_count = int((y == 0).sum().iloc[0])
    minority_count = int((y == 1).sum().iloc[0])
    total_count = majority_count + minority_count

    imbalance_ratio = minority_count / majority_count
    majority_pct = majority_count / total_count * 100
    minority_pct = minority_count / total_count * 100

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = TRAIN_TEST_RATIO, random_state = SEED_GEN, stratify = y)
    
    X_train = X_train.reset_index(drop = True); X_test = X_test.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True); y_test = y_test.reset_index(drop = True)

    X = pd.concat([X_train, X_test], ignore_index = True)
    y = pd.concat([y_train, y_test], ignore_index = True)
    
    X, pipeline = preprocessing(X, conf); num_features = X.shape[1]

    print(f"[INFO] number of input features: {num_features} | majority class samples: {majority_count} ({majority_pct:.1f}%) | " f"minority class samples: {minority_count} ({minority_pct:.1f}%) | "
          f"imbalance ratio (minority/majority): {imbalance_ratio:.3f} | desired balance ratio: {BALANCE_RATIO}")

    raw_imp = compute_importance("f-score_ANOVA", X, y.values.ravel())
    
    X_train = X.iloc[:len(X_train)].reset_index(drop = True)
    X_test  = X.iloc[len(X_train):].reset_index(drop = True)

    gen_params = run_config["gen_params"]
    
    gen_conf = {
        "method": GENERATOR.lower(),
        "params": {
            **gen_params.get(GENERATOR, {}),
            **({"categorical_features": cat_cols} if GENERATOR == "SMOTENC" 
            else {"discrete_features": cat_cols} if GENERATOR in ["CTGAN", "TVAE"] 
            else {"metadata": construct_metadata(conf)} if GENERATOR == "GaussianCopula"
            else {})
        }
    }

    if not WITHOUT_KIPO:
        kipo = KIPO(num_features, tau_threshold = TAU_THRESHOLD, topk_ordering = TOPK_ORDERING, topk_overlap = TOPK_OVERLAP)

        kipo_X_aug, kipo_y_aug, info = kipo.select(X_train, y_train, X_test, y_test, ratio = BALANCE_RATIO,
                                            generator = gen_conf["method"], preprocessing = pipeline, **gen_conf["params"])

        print("[INFO] K-IPO augmentation completed")
        print(f"[INFO] achieved kendall's tau: {info['achieved_kendall_tau']:.4f} | generated samples: {info['minority_class_samples_after'] - info['minority_class_samples_before']}")

    else:
        major = int((y_train == 0).sum().iloc[0]); minor = int((y_train == 1).sum().iloc[0])
        total_needed_samples = max(0, int(major * BALANCE_RATIO) - minor)

        raw_gen_X_aug, raw_gen_y_aug = generate_data_pool(gen_conf["method"], X_train, y_train, total_needed_samples, kipo = False, **gen_conf["params"])

        print("[INFO] raw augmentation completed")

        raw_aug_imp = compute_importance(mode = "f-score_ANOVA", X_train = raw_gen_X_aug, y_train = raw_gen_y_aug.values.ravel())  
        raw_kendall_t = kendall_tau(raw_imp, raw_aug_imp)
        
        print(f"[INFO] achieved kendall's tau: {raw_kendall_t:.4f} | generated samples: {raw_gen_y_aug.shape[0] - minor - (raw_gen_y_aug == 0).sum().iloc[0]}")
        

if __name__ == "__main__":
    main()