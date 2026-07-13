import numpy as np


"""
Gini Index

References:
    Li, J., Cheng, K., Wang, S., Morstatter, F., Trevino, R. P.,
    Tang, J., Liu, H. (2018). Feature Selection: A Data Perspective.
    ACM Computing Surveys (CSUR), 50(6), Article 94.

Source:
    Adapted from the scikit-feature library: https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/statistical_based/gini_index.py
"""


def gini_index(X, y):
    n_features = X.shape[1]

    gini = np.ones(n_features) * 0.5

    for i in range(n_features):
        v = np.unique(X[:, i])
        for j in range(len(v)):
            left_y = y[X[:, i] <= v[j]]; right_y = y[X[:, i] > v[j]]

            gini_left = 0; gini_right = 0

            for k in range(np.min(y), np.max(y)+1):
                if len(left_y) != 0:
                    t1_left = np.true_divide(len(left_y[left_y == k]), len(left_y))
                    t2_left = np.power(t1_left, 2)
                    
                    gini_left += t2_left

                if len(right_y) != 0:
                    t1_right = np.true_divide(len(right_y[right_y == k]), len(right_y))
                    t2_right = np.power(t1_right, 2)
                    
                    gini_right += t2_right

            gini_left = 1 - gini_left; gini_right = 1 - gini_right

            t1_gini = (len(left_y) * gini_left + len(right_y) * gini_right)

            value = np.true_divide(t1_gini, len(y))

            if value < gini[i]:
                gini[i] = value
    
    return gini