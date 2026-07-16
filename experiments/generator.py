import logging; import warnings
import os; import yaml; import pandas as pd; import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

from rich.console import Console

from kipo.selector import KIPOSelector as KIPO
from kipo.utils import load_data, preprocessing, encode_target, extract_target_var, construct_metadata
from kipo.models import models

from utils import (compute_explanations, compute_final_results, calculate_threshold, compute_metrics, 
                   separability_N3, format_feature_importance, compute_importance)

import torcpy

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("smote_variants").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


def work(index, model_name, kipo_X_train_aug, kipo_X_test_aug, kipo_y_train_aug, 
         kipo_y_test_aug, num_cols, cat_cols, xai_methods):
    
    model_conf = models[model_name]
    estimator = model_conf["model"](**model_conf["params"])
    estimator.fit(kipo_X_train_aug, kipo_y_train_aug.values.ravel())

    probs = estimator.predict_proba(kipo_X_test_aug)[:, -1]
    preds = probs >= calculate_threshold(kipo_y_test_aug, probs, criterion = "geo_mean")

    pred_metrics = compute_metrics(probs, preds, kipo_y_test_aug)

    xai_metrics = compute_explanations(X_train = kipo_X_train_aug, y_train = kipo_y_train_aug.values.ravel(), estimator = estimator,
                                        X_test = kipo_X_test_aug, y_test = kipo_y_test_aug.values.ravel(), num_cols = num_cols, 
                                        cat_cols = cat_cols, mode = xai_methods[index])

    return pred_metrics, xai_metrics


def main():
    
    console = Console() 

    console.print(f"[magenta][INFO] Running with {torcpy.num_workers()} torcpy workers.[/magenta]")

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
    MODELS = run_config["models"]
    XAI_METHODS = run_config["XAI_methods"]

    with open(os.path.join(os.path.dirname(__file__), "..", "datasets", "config", f"{os.path.splitext(DATASET)[0]}.yml")) as f:
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
    

    data = load_data(os.path.join(os.path.dirname(__file__), "..", "datasets", "data", DATASET), conf)
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

    start_time = torcpy.gettime()

    kipo = KIPO(num_features, tau_threshold = TAU_THRESHOLD, topk_ordering = TOPK_ORDERING, topk_overlap = TOPK_OVERLAP)

    kipo_X_aug, kipo_y_aug, info = kipo.select(X_train, y_train, X_test, y_test, ratio = BALANCE_RATIO,
                                            generator = gen_conf["method"], preprocessing = pipeline, **gen_conf["params"])
    
    kipo_imp = compute_importance(X_train = kipo_X_aug, y_train = kipo_y_aug.values.ravel(),
                                    num_cols = num_cols, cat_cols = cat_cols, preprocessing = pipeline, mode = "f-score_ANOVA")
    

    print("[INFO] K-IPO augmentation completed")
    print(f"[INFO] achieved kendall's tau: {info['achieved_kendall_tau']:.4f} | generated samples: {info['minority_class_samples_after'] - info['minority_class_samples_before']}")
    
    kipo_X_train_aug, kipo_X_test_aug, kipo_y_train_aug, kipo_y_test_aug = train_test_split(kipo_X_aug, kipo_y_aug, train_size = TRAIN_TEST_RATIO,
                                                                                            random_state = SEED_GEN, stratify = kipo_y_aug)        

    tasks = []
    
    for i, mod_name in enumerate(MODELS):
        task = torcpy.submit(work, i, mod_name, kipo_X_train_aug, kipo_X_test_aug, kipo_y_train_aug, 
                                kipo_y_test_aug, num_cols, cat_cols, XAI_METHODS)
        tasks.append(task)
    
    torcpy.wait()

    pred_results = []; xai_results = []

    for task in tasks:
        metrics, xai = task.result()
        pred_results.append(metrics); xai_results.append(xai)

    final_pred_results = compute_final_results(pred_results)
    final_importance = format_feature_importance(xai_results, XAI_METHODS)

    MI = mutual_info_classif(kipo_X_train_aug, kipo_y_train_aug.values.ravel())
    information_gain = (MI / np.log(2.0)).mean()
    separability = separability_N3(kipo_X_train_aug, kipo_y_train_aug.values.ravel()) # 1 - N3

    console.print(f"[green][INFO] training and testing of all models completed successfully[/green]")

    metrics_form = " | ".join(f"[bold]{k}[/bold]: [bright_cyan]{v:.4f}[/bright_cyan]" for k, v in final_pred_results.items())
    console.print(f"[green][INFO] predictive capabilities -> [/green] {metrics_form}")

    xai_methods = final_importance.columns.tolist()

    console.print(f"[dark_orange][INFO] feature importance ranking:[/dark_orange]")
    for method in xai_methods:
        print(f"{method} -> {final_importance[method].tolist()}")
        
    print(f"[INFO] information gain: {information_gain:.4f}")
    print(f"[INFO] separability (1 - N3): {separability:.4f}")

    end_time = torcpy.gettime()

    console.print(f"[magenta][INFO] total time: {end_time - start_time:.4f}[/magenta]")


if __name__ == "__main__":
    torcpy.start(main)