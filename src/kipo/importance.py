from sklearn.feature_selection import f_classif; from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier; from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier; from xgboost import XGBClassifier 

from catboost import CatBoostClassifier; from lightgbm import LGBMClassifier

import numpy as np
np.int = int

from piml.models import GAMINetClassifier
from piml import Experiment

import contextlib; import io

from scipy.stats import kendalltau
import pandas as pd; import numpy as np

import shap
import torch

from .sofi import SOFI_Explainer


use_gpu = torch.cuda.is_available()
    

def compute_bgshap_dataset(X, size = None):
    if size == None or size >= X.shape[0]:
        return X
    try:
        return pd.DataFrame(shap.kmeans(X, size))
    
    except Exception:
        return pd.DataFrame(shap.sample(X, size))
    

def shap_explainer(est, data):
    match est:
        case DecisionTreeClassifier() | RandomForestClassifier():
            return shap.TreeExplainer(est, feature_perturbation = "tree_path_dependent") 
        
        case XGBClassifier() | CatBoostClassifier() | LGBMClassifier():
            if use_gpu:
                return shap.GPUTreeExplainer(est, feature_perturbation = "tree_path_dependent")
            else:
                return shap.TreeExplainer(est, feature_perturbation = "tree_path_dependent")
        
        case LogisticRegression():
            data = np.array(data)
            return shap.LinearExplainer(est, masker = shap.maskers.Independent(data)) 
            
        case _ if isinstance(est, MLPClassifier):
            return shap.KernelExplainer(est.predict_proba, data)
        
        case _:
            return shap.Explainer(est, data)
        
        
def kendall_tau(a: pd.Series, b: pd.Series):
    common = a.index.intersection(b.index)
    
    ar = a[common].rank(ascending = False, method = 'average')
    br = b[common].rank(ascending = False, method = 'average')

    tau, _ = kendalltau(ar, br)

    return float(tau)


def topk_mag(a: pd.Series, b: pd.Series, k: int) -> bool:
    k -= 1

    rat_ref = a.values[1:] / a.values[:-1] 
    rat_aug = b.values[1:] / b.values[:-1]

    return np.abs(rat_aug[:k]) / np.abs(rat_ref[:k] + 1e-10)


def topk_ordering(a: pd.Series, b: pd.Series, k: int) -> float:
        return a.index[:k].equals(b.index[:k])


def topk_overlap(a: pd.Series, b: pd.Series, k: int) -> bool:
    top_a = set(a.index[:k]); top_b = set(b.index[:k])
    
    return len(top_a & top_b) == k


def avg_importances(importances):
    importances = pd.concat(importances, axis = 1)
    
    return importances.mean(axis = 1).sort_values(ascending = False)


def compute_importance(mode, X_train, y_train, X_test = None, y_test = None, estimator = None, **imp_args):
    
    match mode:
        
        case "estimator_feature_importance":
            if isinstance(estimator, XGBClassifier):
                imp = estimator.get_booster().get_score(importance_type = 'gain') 
            
            elif hasattr(estimator, "feature_importances_"):
                imp = estimator.feature_importances_
            
            elif isinstance(estimator, LogisticRegression):
                imp = estimator.coef_.flatten()
            
            elif isinstance(estimator, CatBoostClassifier):
                imp = estimator.get_feature_importance(type = "FeatureImportance")
          
            elif isinstance(estimator, (GAMINetClassifier)):
                exp = Experiment()
                
                est = exp.make_pipeline(
                    model = estimator,
                    train_x = X_train,
                    train_y = y_train.ravel(),
                    test_x = X_test,
                    test_y = y_test.ravel()
                )

                with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
          
                    from IPython.display import clear_output
                    _ = exp.register(est, "piml_model")
                    clear_output(wait = False)

                features_importance = exp.model_interpret(model = "piml_model", show = "global_fi",  return_data = True).data

                cols_mapping = [X_test.columns[int(f[1:])] if f.startswith("X") and f[1:].isdigit() else f 
                                for f in features_importance["Feature Name"]]
               
                importances = pd.Series(features_importance["Importance"].values, index = cols_mapping).sort_values(ascending = False)
                
                return importances
                            
            else:
                raise ValueError("Estimator does not provide feature importances.")
            
            importances = pd.Series(imp, index = X_test.columns).sort_values(ascending = False)
            
            return importances
        
        case "f-score_ANOVA":
            f_vals = np.nan_to_num(f_classif(X_train, y_train)[0], nan = 0.0, posinf = 0.0, neginf = 0.0)
            
            importances = pd.Series(f_vals, index = X_train.columns).sort_values(ascending = False)
            
            return importances

        case "shap":
            X = compute_bgshap_dataset(X_train, imp_args.get("bg_data_size", 200)); explainer = shap_explainer(estimator, X)
            
            try:
                shap_vals = explainer(X_test).values; features = explainer(X_test).feature_names

            except Exception:
                shap_vals = explainer(X_test, check_additivity = False).values; features = explainer(X_test, check_additivity = False).feature_names

            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]

            importances = pd.Series(np.abs(shap_vals).mean(0), index = features).sort_values(ascending = False)
            
            return importances

        case "sofi":
            sofi = SOFI_Explainer(estimator, X_test, y_test, encoder = None, priors = 'greedy')

            indices, _, _ = sofi.run()
            
            importances = [None] * len(indices)
            for src_idx, dst_idx in enumerate(indices):
                importances[dst_idx] = X_test.columns[src_idx]

            return importances

        case _:
            raise ValueError(f"[ERROR] Unknown importance mode: {mode}. Supported modes are: estimator_feature_importance, f-score_ANOVA, shap, sofi.")