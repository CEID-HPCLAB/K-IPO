from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import psutil; import math
cores = psutil.cpu_count(logical = False)

import numpy as np
np.int = int

from piml.models import GAMINetClassifier


models = {
    "RandomForest": {
        "model": RandomForestClassifier,
        "params": {"n_estimators": 100}
    },
    "MLP": {
        "model": MLPClassifier,
        "params": {"early_stopping": False}
    },
    "GAMINet": {
        "model": GAMINetClassifier,
        "params": {
            "n_jobs": math.ceil(cores / 3)
        }
    }
}