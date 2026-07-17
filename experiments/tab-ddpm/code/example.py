import os; import yaml
from train import train; from sample import sample
from eval_simple import train_simple; import pandas as pd
import lib; import numpy as np; import torch; from time import time
from sklearn.model_selection import train_test_split

from kipo.utils import load_data, preprocessing, encode_target, extract_target_var
from kipo.importance import compute_importance, kendall_tau


def accept(chunk, X_train, y_train, raw_imp, threshold, data_config):
    aug_imp = compute_importance(mode = "f-score_ANOVA", X_train = preprocessing(pd.concat([X_train, chunk], ignore_index = True), data_config)[0],
                                y_train = pd.concat([y_train, pd.DataFrame({y_train.columns[0]: [1] * chunk.shape[0]})], 
                                ignore_index = True).to_numpy().ravel()) 
    
    if kendall_tau(raw_imp, aug_imp) >= threshold:
        return True

    return False


def load_config(path):
    with open(os.path.join(os.path.dirname(__file__), path)) as f:
        return yaml.safe_load(f)
        

def main():
    
    run_config = load_config(os.path.join("..", "config.yml"))
    DATASET = run_config["dataset"] 
    
    data_config = load_config(os.path.join("..", "..", "..", "datasets", "config", f"{DATASET}.yml"))

    CONFIG = os.path.join(os.path.dirname(__file__), "..", "datasets", "config", f"{DATASET}.toml")
    TAU_THRESHOLD = run_config["tau_threshold"]; BALANCE_RATIO = run_config["balance_ratio"]
    SEED = run_config["seed"]; TRAIN_TEST_RATIO = run_config["train_test_ratio"]

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
    

    max_block_size = min(max(total_needed_samples // 4, 2500), total_needed_samples); min_block_size =  min(int(total_needed_samples * 0.05), 50)


    col_to_idx = {col: i for i, col in enumerate(num_cols + cat_cols)}; reorder_idx = [col_to_idx[col] for col in X.columns]
   
    max_dry_attempts = 25
    
    while total_needed_samples > 0: 
        current_seed = seed_generator.integers(0, 2**31 - 1).item()        
        block_size = min(max_block_size, total_needed_samples); min_block_size = min(min_block_size, block_size)

        train(parent_dir = raw_config['parent_dir'], **raw_config['train']['main'], **raw_config['diffusion_params'], 
            real_data_path = raw_config['real_data_path'], model_type = raw_config['model_type'],
            model_params = raw_config['model_params'], T_dict = raw_config['train']['T'], 
            num_numerical_features = raw_config['num_numerical_features'], device = torch.device(raw_config['device']), 
            change_val = False, seed = current_seed)

        X_num_gen, X_cat_gen, y_gen = sample(num_samples = block_size, parent_dir=raw_config['parent_dir'],
            **raw_config['diffusion_params'], real_data_path = raw_config['real_data_path'], 
            model_path = os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_type = raw_config['model_type'], model_params = raw_config['model_params'],
            T_dict = raw_config['train']['T'], num_numerical_features = raw_config['num_numerical_features'],
            device = torch.device(raw_config['device']), seed = current_seed, disbalance = "fill")
            
        if X_num_gen is not None and X_cat_gen is not None:
            X_aug = np.concatenate([X_num_gen, X_cat_gen], axis = 1)
            
            X_num_real = np.load(os.path.join(raw_config['real_data_path'], "X_num_train.npy"), allow_pickle = True)
            X_num_real = pd.DataFrame(X_num_real, columns = num_cols)
            
            X_cat_real = np.load(os.path.join(raw_config['real_data_path'], "X_cat_train.npy"), allow_pickle = True)
            X_cat_real = pd.DataFrame(X_cat_real, columns = cat_cols)
            
            X_real = pd.concat([X_num_real, X_cat_real], axis = 1)

        elif X_num_gen is not None:
            X_aug = X_num_gen
            
            X_num_real = np.load(os.path.join(raw_config['real_data_path'], "X_num_train.npy"), allow_pickle = True)
            X_num_real = pd.DataFrame(X_num_real, columns = num_cols)
            
            X_real = X_num_real
        else:
            X_aug = X_cat_gen

            X_cat_real = np.load(os.path.join(raw_config['real_data_path'], "X_cat_train.npy"), allow_pickle = True)
            X_cat_real = pd.DataFrame(X_cat_real, columns = cat_cols)
            
            X_real = X_cat_real

        X_aug = X_aug[:, reorder_idx]
        X_aug_df = pd.DataFrame(X_aug, columns = X.columns)
            
        if accept(X_aug_df, X_real, y_train, raw_imp, TAU_THRESHOLD, data_config):
            total_needed_samples -= X_aug_df.shape[0]; X_real = pd.concat([X_real, X_aug_df], ignore_index = True)
            y_train = pd.concat([y_train, pd.DataFrame({y_train.columns[0]: [1] * X_aug_df.shape[0]})], ignore_index = True)

            np.save(os.path.join(os.path.dirname(__file__), "..", "datasets", "data", DATASET, "y_train.npy"), y_train.values.reshape(-1))
            if X_num_gen is not None and X_cat_gen is not None:
                x_num_train = X_real[num_cols]; x_cat_train = X_real[cat_cols]
                x_num_train = X_real[num_cols].astype(np.float32)
                
                np.save(os.path.join(os.path.dirname(__file__), "..", "datasets", "data", DATASET, "X_num_train.npy"), x_num_train.values)
                np.save(os.path.join(os.path.dirname(__file__), "..", "datasets", "data", DATASET, "X_cat_train.npy"), x_cat_train.values)
                

            elif X_num_gen is None:
                np.save(os.path.join(os.path.dirname(__file__), "..", "datasets", "data", DATASET, "X_cat_train.npy"), X_real.values)
                
            else:
                x_num_train = X_real.astype(np.float32)
                np.save(os.path.join(os.path.dirname(__file__), "..", "datasets", "data", DATASET, "X_num_train.npy"), X_real.values)

            continue
            
        i = 1;  dry_attempts = 0
        
        while (X_aug_df.shape[0] // 2**i >= min_block_size):
            
            chunk_size = X_aug_df.shape[0] // 2**i
            drop_idxs = []; accepted_any = False

            for start_idx in range(0, X_aug_df.shape[0], chunk_size):
                end_idx = min(start_idx + chunk_size, X_aug_df.shape[0])
                chunk = X_aug_df.iloc[start_idx: end_idx]

                if chunk.shape[0] < min_block_size:
                    break
                    
                if accept(chunk, X_real, y_train, raw_imp, TAU_THRESHOLD, data_config):
                    total_needed_samples -= chunk.shape[0]; X_real = pd.concat([X_real, chunk], ignore_index = True)
                    y_train = pd.concat([y_train, pd.DataFrame({y_train.columns[0]: [1] * chunk.shape[0]})], ignore_index = True)

                    np.save(os.path.join(os.path.dirname(__file__), "..", "datasets", "data", DATASET, "y_train.npy"), y_train.values.reshape(-1))
                    if X_num_gen is not None and X_cat_gen is not None:
                        x_num_train = X_real[num_cols]; x_cat_train = X_real[cat_cols]
                        x_num_train = X_real[num_cols].astype(np.float32)
                        
                        np.save(os.path.join(os.path.dirname(__file__), "..", "datasets", "data", DATASET, "X_num_train.npy"), x_num_train.values)
                        np.save(os.path.join(os.path.dirname(__file__), "..", "datasets", "data", DATASET, "X_cat_train.npy"), x_cat_train.values)
                        

                    elif X_num_gen is None:
                        np.save(os.path.join(os.path.dirname(__file__), "..", "datasets", "data", DATASET, "X_cat_train.npy"), X_real.values)
                        
                    else:
                        x_num_train = X_real.astype(np.float32)
                        np.save(os.path.join(os.path.dirname(__file__), "..", "datasets", "data", DATASET, "X_num_train.npy"), X_real.values)
                            
                    drop_idxs.extend(range(start_idx, end_idx)); accepted_any = True; dry_attempts = 0
            
                else:
                    dry_attempts += 1
                    if dry_attempts >= max_dry_attempts:
                        break
 
        
            if dry_attempts >= max_dry_attempts:
                break  

            if not accepted_any:
                pass

            else:
                X_aug_df = X_aug_df.drop(drop_idxs); X_aug_df.reset_index(drop = True, inplace = True)
                
            i += 1

    y_aug = encode_target(y_train, data_config)
    X_aug, _ = preprocessing(X_real, data_config)

    raw_aug_imp = compute_importance(mode = "f-score_ANOVA", X_train = X_aug, y_train = y_aug.values.ravel())  
    raw_kendall_t = kendall_tau(raw_imp, raw_aug_imp)
        
    print(f"[INFO] achieved kendall's tau: {raw_kendall_t:.4f} | generated samples: {y_aug.shape[0] - (y_aug == 0).sum().iloc[0]}")

    

if __name__ == '__main__':
    main()