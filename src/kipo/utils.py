import pandas as pd; import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer

from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score, 
                             matthews_corrcoef, balanced_accuracy_score, brier_score_loss, 
                             accuracy_score, precision_score, recall_score, roc_curve, 
                             precision_recall_curve)

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression; from sklearn.neighbors import NearestNeighbors

from sdv.metadata import Metadata


def load_data(file_path, conf):
    file_path = file_path if file_path.endswith(".csv") else file_path + ".csv"
    
    return pd.read_csv(file_path, sep = conf.get("sep", ","))


def init_preprocessing_pipeline(conf):
    drop_cols = conf.get("drop_cols", []); num_cols = conf.get("num_cols", []); cat_cols_conf = conf.get("cat_cols", [])

    cat_cols = [list(col.keys())[0] for col in cat_cols_conf]

    cat_orders = {list(col.keys())[0]: col.get("order", None) for col in cat_cols_conf}
    ordinal_cols = [col for col in cat_cols if cat_orders[col] is not None]
    ohe_cols = [col for col in cat_cols if cat_orders[col] is None]

    dropper = FunctionTransformer(lambda df: df.drop([c for c in drop_cols if c in df.columns], axis = 1))

    transformers = []

    if ordinal_cols:
        categories = [cat_orders[col] for col in ordinal_cols]  
        ordinal_pipeline = Pipeline([
            ("ordinal_encoder", OrdinalEncoder(categories = categories, dtype = float)),
            ("scaler", StandardScaler())
        ])
        transformers.append(("ordinal_cat", ordinal_pipeline, ordinal_cols))

    if ohe_cols:
        ohe_pipeline = Pipeline([
            ("ordinal_encoder", OrdinalEncoder(dtype = float)),
            ("scaler", StandardScaler())
        ])
        transformers.append(("ohe_cat", ohe_pipeline, ohe_cols))

    if num_cols:
        num_pipeline = Pipeline([
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, num_cols))

    preprocessor = ColumnTransformer(
        transformers = transformers,
        remainder = "passthrough"
    )

    return Pipeline([
        ("drop_cols", dropper),
        ("preprocessor", preprocessor)
    ])


def extract_target_var(data, conf):
    target_info = conf.get("target_col", {}); target_name = target_info.get("name", "target")
    X = data.drop([target_name], axis = 1); y = data[target_name]

    return X, y


def encode_target(y, conf):
    target_info = conf.get("target_col", {}); target_name = target_info.get("name", "target")
    
    if not target_info.get("encoding", True):
        y_series = pd.Series(y); counts = y_series.value_counts()
        majority_class = counts.idxmax(); minority_class = counts.idxmin()  
        
        mapping = {majority_class: 0, minority_class: 1}
        y_encoded = y_series.map(mapping)
        
        return pd.DataFrame(y_encoded, columns = [target_name])
    
    return pd.DataFrame(y, columns = [target_name])


def preprocessing(data, conf):
    pipeline = init_preprocessing_pipeline(conf)
    X_transformed = pipeline.fit_transform(data)

    preprocessors = pipeline.named_steps["preprocessor"]
    features = [name.split("__")[-1] for name in preprocessors.get_feature_names_out()]

    return pd.DataFrame(X_transformed, columns = features), pipeline


def inverse_transform(data, pipeline):
    preprocessors = pipeline.named_steps["preprocessor"]

    raw_data = pd.DataFrame()

    for _, operator, cols in preprocessors.transformers_:
        if isinstance(operator, str): # passthrough
            raw_data = pd.concat([raw_data, data[preprocessors.feature_names_in_[list(cols)]]], axis = 1)
            continue
        
        if not hasattr(operator, "inverse_transform"):
            raise ValueError("The preprocessor does not support inverse_transform")
        
        raw_data = pd.concat([raw_data, pd.DataFrame(operator.inverse_transform(data[operator.get_feature_names_out()]), columns = cols)], axis = 1)

    return raw_data


def compute_metrics(probs, preds, y_test):
     return {
        'Accuracy': float(accuracy_score(y_test, preds)), 'Precision': float(precision_score(y_test, preds)),
        'F1': float(f1_score(y_test, preds)), 'Recall': float(recall_score(y_test, preds)), 'ROC_AUC': float(roc_auc_score(y_test, probs)),
        'PR_AUC': float(average_precision_score(y_test, probs)), 'MCC': float(matthews_corrcoef(y_test, preds)),
        'BalancedAcc': float(balanced_accuracy_score(y_test, preds)), 'Brier': float(brier_score_loss(y_test, probs)),
     }


def linear_separability(X, y):
    clf = LogisticRegression(); scores = cross_val_score(clf, X, y)
    linear_sep = scores.mean() 
    
    return linear_sep


def separability_N3(X, y):
    X_np = np.array(X); y_np = np.array(y)
    
    nn = NearestNeighbors(); nn.fit(X_np); _, indices = nn.kneighbors(X_np)
    nn_labels = y_np[indices[:, 1]]; n3 = np.mean(nn_labels != y_np)
    
    return 1 - n3, n3


def construct_metadata(conf, table_name = "data"):
    (metadata := Metadata()).add_table(table_name)

    for n_col in conf.get('num_cols', []):
            metadata.add_column(n_col, sdtype = 'numerical')

    for c_col in conf.get('cat_cols', []):
            metadata.add_column(c_col, sdtype = 'categorical')

    return metadata


def calculate_threshold(labels, probs, criterion = "geo_mean", cost_matrix = None):
    fpr, tpr, thresholds = roc_curve(labels, probs, pos_label = 1, drop_intermediate = False)
    
    match criterion:      
        case "geo_mean":
            gmeans = np.sqrt(tpr * (1 - fpr))
            optimal_idx = np.argmax(gmeans)
            return thresholds[optimal_idx]
        
        case "roc":
            cost_matrix = np.array([[0., 1], [1, 0.]]) if cost_matrix is None else cost_matrix

            fpr = np.asarray(fpr)
            tpr = np.asarray(tpr)

            slope = ((cost_matrix[1, 0] - cost_matrix[1,1]) /
                    (cost_matrix[0, 1] - cost_matrix[0,0])) \
                    * ((labels == 1.0).sum() / (labels == 0.0).sum())

            slope = float(slope)  

            opt_yl = 1 - slope * fpr
            dist = np.abs(tpr - opt_yl / slope)
            optimal_idx = np.argmin(dist)

            return thresholds[optimal_idx]
            
        case "Youden J":
            J = tpr - fpr
            optimal_idx = np.argmax(J)
            return thresholds[optimal_idx]
        
        case "f1":
            precision, recall, thresholds = precision_recall_curve(labels, probs)
            f1 = np.zeros_like(precision)  
            valid_mask = (precision + recall) > 0  
            f1[valid_mask] = 2 * (precision[valid_mask] * recall[valid_mask]) / (precision[valid_mask] + recall[valid_mask])
            optimal_idx = np.argmax(f1)
            return thresholds[optimal_idx]

        case _:
            raise ValueError("Invalid criterion. Supported criteria are 'geo_mean', 'roc', 'Youden J', and 'f1'.")


def construct_metadata(conf, table_name = "metadata"):
    cat_cols = [list(col.keys())[0] for col in conf.get("cat_cols", [])]; num_cols = conf.get("num_cols", [])

    (metadata := Metadata()).add_table(table_name)

    for num_col in num_cols:
            metadata.add_column(num_col, sdtype = 'numerical')

    for cat_col in cat_cols:
            metadata.add_column(cat_col, sdtype = 'categorical')

    return metadata