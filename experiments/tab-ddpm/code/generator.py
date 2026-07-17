import os; import yaml
from train import train; from sample import sample
import pandas as pd
import lib; import numpy as np; import torch; from time import time
from sklearn.model_selection import train_test_split

from kipo.utils import load_data, preprocessing, encode_target, extract_target_var
from kipo.importance import compute_importance, kendall_tau


def load_config(path):
    with open(os.path.join(os.path.dirname(__file__), path)) as f:
        return yaml.safe_load(f)
        

def main():
    
    run_config = load_config(os.path.join("..", "config.yml"))
    DATASET = run_config["dataset"] 
    
    data_config = load_config(os.path.join("..", "..", "..", "datasets", "config", f"{DATASET}.yml"))

    CONFIG = os.path.join(os.path.dirname(__file__), "..", "datasets", "config", f"{DATASET}.toml")
    BALANCE_RATIO = run_config["balance_ratio"]; SEED = run_config["seed"]; TRAIN_TEST_RATIO = run_config["train_test_ratio"]

    raw_config = lib.load_config(CONFIG)
    
    num_cols = data_config.get("num_cols", [])

    cat_cols_config = data_config.get("cat_cols", [])
    cat_cols = [list(cat.keys())[0] for cat in cat_cols_config] if cat_cols_config else []    

    data = load_data(os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets", "data", DATASET), data_config)
    X, y = extract_target_var(data, data_config)
    
    y = encode_target(y, data_config)

    missing_num = set(num_cols) - set(X.columns); missing_cat = set(cat_cols) - set(X.columns)

    if missing_num:
        raise ValueError(f"Missing numerical columns: {missing_num}")

    if missing_cat:
        raise ValueError(f"Missing categorical columns: {missing_cat}")
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = TRAIN_TEST_RATIO, random_state = SEED, stratify = y)
    
    X_train = X_train.reset_index(drop = True); X_test = X_test.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True); y_test = y_test.reset_index(drop = True)

    X = pd.concat([X_train, X_test], ignore_index = True); y = pd.concat([y_train, y_test], ignore_index = True)
    
    X, _ = preprocessing(X, data_config)
    raw_imp = compute_importance(mode = "f-score_ANOVA", X_train = X, y_train = y.values.ravel()) 
    
    major = int((y_train == 0).sum().iloc[0]); minor = int((y_train == 1).sum().iloc[0])
    total_needed_samples = max(0, int(major * BALANCE_RATIO) - minor)

    seed_generator = np.random.default_rng(int(time())); current_seed = seed_generator.integers(0, 2**31 - 1).item()

    train(parent_dir = raw_config['parent_dir'], **raw_config['train']['main'], **raw_config['diffusion_params'], 
         real_data_path = raw_config['real_data_path'], model_type = raw_config['model_type'],
         model_params = raw_config['model_params'], T_dict = raw_config['train']['T'], 
         num_numerical_features = raw_config['num_numerical_features'], device = torch.device(raw_config['device']), 
         change_val = False, seed = current_seed)
    
    X_num_gen, X_cat_gen, y_gen = sample(num_samples = total_needed_samples, parent_dir = raw_config['parent_dir'],
        balance_ratio = BALANCE_RATIO,
        **raw_config['diffusion_params'], real_data_path = raw_config['real_data_path'], 
        model_path = os.path.join(raw_config['parent_dir'], 'model.pt'),
        model_type = raw_config['model_type'], model_params = raw_config['model_params'],
        T_dict = raw_config['train']['T'], num_numerical_features = raw_config['num_numerical_features'],
        device = torch.device(raw_config['device']), seed = current_seed)
    
    if X_num_gen is not None and X_cat_gen is not None:
        X_gen = np.concatenate([X_num_gen, X_cat_gen], axis = 1)

    elif X_num_gen is not None:
        X_gen = X_num_gen

    else:
        X_gen = X_cat_gen
    
    print("[INFO] TabDDPM augmentation completed")

    col_to_idx = {col: i for i, col in enumerate(num_cols + cat_cols)}; reorder_idx = [col_to_idx[col] for col in X.columns]
    X_gen = X_gen[:, reorder_idx]
    
    y_gen = pd.DataFrame(y_gen, columns = [data_config["target_col"]["name"]])
    X_gen = pd.DataFrame(X_gen, columns = X.columns)

    X_aug = pd.concat([X_train, X_gen], ignore_index = True); y_aug = pd.concat([y_train.squeeze(), y_gen.squeeze()], ignore_index = True)

    y_aug = encode_target(y_aug, data_config)
    X_aug, _ = preprocessing(X_aug, data_config)

    raw_aug_imp = compute_importance(mode = "f-score_ANOVA", X_train = X_aug, y_train = y_aug.values.ravel())  
    raw_kendall_t = kendall_tau(raw_imp, raw_aug_imp)
        
    print(f"[INFO] achieved kendall's tau: {raw_kendall_t:.4f} | generated samples: {y_aug.shape[0] - (y_aug == 0).sum().iloc[0]}")
        

if __name__ == '__main__':
    main()