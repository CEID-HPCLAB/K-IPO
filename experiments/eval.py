import logging; import warnings
import os; import yaml; import pandas as pd; import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

from rich.console import Console

from kipo.selector import KIPOSelector as KIPO
from kipo.utils import load_data, preprocessing, encode_target, extract_target_var
from kipo.models import models
from kipo.importance import compute_importance, kendall_tau

import copy

from utils import (compute_explanations, compute_final_results, calculate_threshold, compute_metrics, 
                   separability_N3, format_feature_importance, compute_importance_overlap,
                   compute_topk_ordering, get_num_topk_features)

import torcpy

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("smote_variants").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


def work(index, model_name, kipo_X_train_aug, kipo_X_test_aug, kipo_y_train_aug, 
         kipo_y_test_aug, num_cols, cat_cols, xai_methods):
    
    model_conf = models[model_name]
    estimator = copy.deepcopy(model_conf["model"](**model_conf["params"]))
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

    n_workers = torcpy.num_workers()
    
    plural = "" if n_workers == 1 else "s"
    console.print(f"[magenta][INFO] running with {n_workers} torcpy worker{plural}.[/magenta]")

    def load_config():
        with open(os.path.join(os.path.dirname(__file__), "config.yml")) as f:
            return yaml.safe_load(f)

    run_config = load_config()

    DATASET = run_config["dataset"]
    EVAL_DATASET = run_config["eval_dataset"]
    TRAIN_TEST_RATIO = run_config["train_test_ratio"]
    SEED_GEN = run_config["seed"]
    MODELS = run_config["models"]
    XAI_METHODS = run_config["XAI_methods"]

    with open(os.path.join(os.path.dirname(__file__), "..", "datasets", "config", f"{os.path.splitext(DATASET)[0]}.yml")) as f:
        conf = yaml.safe_load(f)

    cat_cols = [list(col.keys())[0] for col in conf.get("cat_cols", [])]; num_cols = conf.get("num_cols", [])    

    data = load_data(os.path.join(os.path.dirname(__file__), "..", "datasets", "data", DATASET), conf)
    X, y = extract_target_var(data, conf)
    
    y = encode_target(y, conf)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = TRAIN_TEST_RATIO, random_state = SEED_GEN, stratify = y)
    
    X_train = X_train.reset_index(drop = True); X_test = X_test.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True); y_test = y_test.reset_index(drop = True)

    X = pd.concat([X_train, X_test], ignore_index = True)
    y = pd.concat([y_train, y_test], ignore_index = True)
    
    X, _ = preprocessing(X, conf)

    raw_imp = compute_importance(mode = "f-score_ANOVA", X_train = X, y_train = y.values.ravel())

    num_topk_features = get_num_topk_features(DATASET)

    start_time = torcpy.gettime()
    
    aug_data = load_data(os.path.join(os.path.dirname(__file__), EVAL_DATASET), conf)
    X_aug = aug_data.iloc[:, :-1]; y_aug = aug_data.iloc[:, -1]

    X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(X_aug, y_aug, train_size = TRAIN_TEST_RATIO,
                                                                        random_state = SEED_GEN, stratify = y_aug)
    
    tasks = []
    
    for i, mod_name in enumerate(MODELS):
        task = torcpy.submit(work, i, mod_name, X_train_aug, X_test_aug, y_train_aug, 
                            y_test_aug, num_cols, cat_cols, XAI_METHODS)
        tasks.append(task)
    
    torcpy.wait()
    
    pred_results = []; xai_results = []

    for task in tasks:
        metrics, xai = task.result()
        pred_results.append(metrics); xai_results.append(xai)

    final_pred_results = compute_final_results(pred_results)
    final_importance = format_feature_importance(xai_results, XAI_METHODS)

    MI = mutual_info_classif(X_train_aug, y_train_aug.values.ravel())
    information_gain = (MI / np.log(2.0)).mean()
    seperability = separability_N3(X_train_aug, y_train_aug.values.ravel())

    console = Console() 
    console.print(f"[green][INFO] training and testing of all models completed successfully[/green]")

    metrics_form = " | ".join(f"[bold]{k}[/bold]: [bright_cyan]{v:.4f}[/bright_cyan]" for k, v in final_pred_results.items())
    console.print(f"[green][INFO] predictive capabilities -> [/green] {metrics_form}")

    print(f"[INFO] information gain: {information_gain:.4f}")
    print(f"[INFO] seperability (1 - N3): {seperability:.4f}")

    importance_overlap_perc = compute_importance_overlap(final_importance, DATASET, topk = num_topk_features) 
    console.print(f"[dark_orange][INFO] feature importance ranking (overlap) ->[/dark_orange] "f"[white]{importance_overlap_perc:.3f} %[/white]")

    aug_imp = compute_importance(mode = "f-score_ANOVA", X_train = X_aug, y_train = y_aug.values.ravel())  
    kendall_t = kendall_tau(raw_imp, aug_imp); topk_ordering = compute_topk_ordering(raw_imp, aug_imp)
        
    print(f"[INFO] kendall's tau: {kendall_t:.4f}, top-k ordering: {topk_ordering}")

    end_time = torcpy.gettime()

    console.print(f"[magenta][INFO] total time: {end_time - start_time:.4f}[/magenta]")


if __name__ == "__main__":
    torcpy.start(main)