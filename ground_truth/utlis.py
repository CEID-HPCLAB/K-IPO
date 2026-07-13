import numpy as np; import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from skrebate import ReliefF

from ITMO_FS.filters.multivariate import JMI, MRI

from laplacian import laplacian_score; from gini import gini_index

import dcor


def mutual_info_ranking(X, y):
    scores = pd.Series(mutual_info_classif(X, y), index = X.columns)
    
    return scores.sort_values(ascending = False).index.tolist()


def laplacian_score_ranking(X, y):
    scores =  pd.Series(laplacian_score(np.asarray(X), y), index = X.columns)
    
    return scores.sort_values(ascending = True).index.tolist()


def reliefF_ranking(X, y):
    relief = ReliefF(n_neighbors = 10, n_features_to_select = X.shape[1])

    relief.fit(X.values, y.values)

    scores = pd.Series(relief.feature_importances_, index = X.columns)

    return scores.sort_values(ascending = False).index.tolist()


def gini_index_ranking(X, y):
    X_np = X.values; y_np = np.array(y).ravel() 
    
    scores = gini_index(X_np, y_np)
    scores = pd.Series(scores, index = X.columns)
    
    return scores.sort_values(ascending = True).index.tolist()


def jmi_ranking(X, y):
    X_np = X.values; y_np = np.array(y).ravel()  
    n_features = X.shape[1]
    
    selected_features = []
    rem_features = list(range(n_features))

    ranking = []
    while rem_features:
        scores = JMI(np.array(selected_features), np.array(rem_features), X_np, y_np)
        best_idx = np.argmax(scores); best_feature = rem_features[best_idx]

        ranking.append(best_feature); selected_features.append(best_feature); rem_features.remove(best_feature)

    return [X.columns[i] for i in ranking]


def mri_ranking(X, y):
    X_np = X.values; y_np = np.array(y).ravel()  
    n_features = X.shape[1]

    selected_features = []
    rem_features = list(range(n_features))

    ranking = []
    while rem_features:
        scores = MRI(np.array(selected_features), np.array(rem_features), X_np, y_np)
        best_idx = np.argmax(scores); best_feature = rem_features[best_idx]

        ranking.append(best_feature); selected_features.append(best_feature); rem_features.remove(best_feature)

    return [X.columns[i] for i in ranking]


def distance_corr_ranking(X, y):
    scores = []

    for i in range(X.shape[1]):
        score = dcor.distance_correlation(X.iloc[:, i], np.asarray(y, dtype = float))
        scores.append(score)
    
    scores = pd.Series(scores, index = X.columns)
    

    return scores.sort_values(ascending = False).index.tolist()


def compute_ground_truth(X, y):
    methods = {"Distance Correlation": distance_corr_ranking, "MI": mutual_info_ranking, "Laplacian Score": laplacian_score_ranking,
        "ReliefF": reliefF_ranking, "Gini Index": gini_index_ranking, "JMI": jmi_ranking, "MRI": mri_ranking
    }

    rankings = {name: method(X, y.squeeze()) for name, method in methods.items()}

    return pd.DataFrame(rankings)