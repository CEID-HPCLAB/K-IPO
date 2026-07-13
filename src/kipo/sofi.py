import numpy as np; import pandas as pd

import random; import warnings; from tqdm import tqdm

from sklearn.metrics import f1_score, auc
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


"""
SOFI: Sparseness-Optimized Feature Importance

References:
    Grau, I., Nápoles, G. (2024). Sparseness-Optimized Feature Importance. 
    In: Longo, L., Lapuschkin, S., Seifert, C. (eds) Explainable Artificial Intelligence. xAI 2024. 
    Communications in Computer and Information Science, vol 2154. Springer, Cham

Source:
    Adapted from the official SOFI repository: https://github.com/igraugar/sofi
"""


class SOFI_Explainer:
    
    def __init__(self, model, X, y, encoder, priors = 'greedy'):

        self.model = model
        self.encoder = encoder

        try:
            check_is_fitted(self.model)
        except NotFittedError:
            pass
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a Pandas DataFrame with correct data types.")

        if not hasattr(self.model, 'feature_names_in_'):
            raise ValueError("Refit the model with a Pandas DataFrame with feature names.")
            
        self.int_indexes = [X.columns.get_loc(col) for col in X.select_dtypes(include = ['int64', 'int32']).columns.tolist()]
        self.float_indexes = [X.columns.get_loc(col) for col in X.select_dtypes(include = ['float64', 'float32']).columns.tolist()]
        self.nominal_indexes = [X.columns.get_loc(col) for col in X.select_dtypes(include = ['object', 'category']).columns.tolist()]
        
        self.numeric_features = X.select_dtypes(include = ['int64', 'int32', 'float64', 'float32']).columns.tolist()
        self.nominal_features = X.select_dtypes(include = ['object', 'category']).columns.tolist()

        has_nominal = len(self.nominal_features) > 0
        if self.encoder is not None:
            if not isinstance(self.encoder, ColumnTransformer):
                raise ValueError("Provided encoder must be an instance of sklearn.compose.ColumnTransformer.")
            if not has_nominal:
                warnings.warn("Encoder provided but no nominal features detected.")
        elif has_nominal:
            raise ValueError("Nominal features detected but no encoder provided.")
        
        if self.encoder is not None:
            X_encoded_arr = self.encoder.transform(X)
            self.feature_names_encoded = self.encoder.get_feature_names_out()
            X_encoded = pd.DataFrame(X_encoded_arr, columns = self.feature_names_encoded, index = X.index)
        else:
            X_encoded = X.copy()
            self.feature_names_encoded = X.columns
        
        y_pred = self.model.predict(X_encoded)
        
        correct_mask = y_pred == y
        self.X_correct = X[correct_mask].copy()
        self.y_correct = y[correct_mask]
        
        if self.encoder is not None:
            X_correct_encoded_arr = self.encoder.transform(self.X_correct)
            self.X_correct_encoded = pd.DataFrame(X_correct_encoded_arr, columns = self.feature_names_encoded, index = self.X_correct.index)
        else:
            self.X_correct_encoded = self.X_correct.copy()
        
        self.marginalized_values = {}
        for feature_idx in range(self.X_correct.shape[1]):
            col = self.X_correct.iloc[:, feature_idx]
            if feature_idx in self.nominal_indexes:
                self.marginalized_values[feature_idx] = col.mode()[0]
            elif feature_idx in self.int_indexes:
                self.marginalized_values[feature_idx] = np.int64(col.mode()[0])
            elif feature_idx in self.float_indexes:
                self.marginalized_values[feature_idx] = np.float64(col.mean())
            else:
                self.marginalized_values[feature_idx] = col.mode()[0]
        
        self.feature_to_encoded = {}
        current_col = 0
        for col_name in self.numeric_features:
            feat_idx = X.columns.get_loc(col_name)
            self.feature_to_encoded[feat_idx] = current_col
            current_col += 1
        
        self.marginalized_encoded = {}
        if has_nominal:
            onehot = self.encoder.named_transformers_['onehot']
            for i, col_name in enumerate(self.nominal_features):
                feat_idx = X.columns.get_loc(col_name)
                n_out = len(onehot.categories_[i]) - 1
                self.feature_to_encoded[feat_idx] = slice(current_col, current_col + n_out)
                current_col += n_out


            for i, feat_idx in enumerate(self.nominal_indexes):
                mode_val = self.marginalized_values[feat_idx]
                cats = onehot.categories_[i]
                n_out = len(cats) - 1
                vec = np.zeros(n_out, dtype = int)
                if mode_val != cats[0]:
                    cat_idx = np.where(cats == mode_val)[0][0]
                    if cat_idx > 0:
                        vec[cat_idx - 1] = 1
                self.marginalized_encoded[feat_idx] = vec
        
        self.n_features = self.X_correct.shape[1]
        
        if priors == 'greedy':
            self.baseline_f1 = f1_score(self.y_correct, self.model.predict(self.X_correct_encoded), average = 'macro')
            self.priors = np.zeros(self.n_features, dtype = np.float64)
            for feature_idx in range(self.n_features):
                encoded_marg = self.X_correct_encoded.copy()
                enc_pos = self.feature_to_encoded[feature_idx]
                if isinstance(enc_pos, int):  
                    marg_val = self.marginalized_values[feature_idx]
                    encoded_marg.iloc[:, enc_pos] = marg_val
                else:  
                    encoded_marg.iloc[:, enc_pos] = self.marginalized_encoded[feature_idx]
                y_pred_marg = self.model.predict(encoded_marg)
                f1_marg = f1_score(self.y_correct, y_pred_marg, average = 'macro')
                drop = self.baseline_f1 - f1_marg
                self.priors[feature_idx] = drop
        else:
            self.priors = np.array(priors, dtype = np.float64)
            if len(self.priors) != self.n_features:
                raise ValueError("Provided priors must have length equal to the number of features.")


    def accumulative_marginalization(self, permutation):

        scores = [1.0] 
        encoded_marginalized = self.X_correct_encoded.copy()
        
        for feature_idx in permutation:
            
            enc_pos = self.feature_to_encoded[feature_idx]
            if isinstance(enc_pos, int):  
                marg_val = self.marginalized_values[feature_idx]
                encoded_marginalized.iloc[:, enc_pos] = marg_val
            else:  
                encoded_marginalized.iloc[:, enc_pos] = self.marginalized_encoded[feature_idx]
            
            y_pred_marginalized = self.model.predict(encoded_marginalized)
            f1 = f1_score(self.y_correct, y_pred_marginalized, average = 'macro')
            scores.append(round(f1, 4))
        
        return scores

    def compute_permutation_auc(self, scores):

        x = np.arange(len(scores))
        y = np.array(scores)
        return auc(x, y)

    def probabilistic_sample_swap(self, current_permutation, tau):

        n = len(current_permutation)
        x = np.array(current_permutation)
        priors_x = self.priors[x]
        
        i_idx, j_idx = np.triu_indices(n, k=1)
        utilities = np.maximum(0, priors_x[j_idx] - priors_x[i_idx])
        
        exp_utilities = np.exp(utilities / tau)
        if exp_utilities.sum() == 0:
            probabilities = np.ones_like(exp_utilities) / len(exp_utilities)
        else:
            probabilities = exp_utilities / exp_utilities.sum()
        
        chosen_idx = np.random.choice(len(probabilities), p=probabilities)
        return i_idx[chosen_idx], j_idx[chosen_idx]

    def run(self, tau = 0.1, max_iterations = 200, learning_rate = 0.05, baseline_decay = 0.9, seed = 512):
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        evaluated_permutations = set()
        
        current_permutation = sorted(range(self.n_features), key=lambda i: self.priors[i], reverse = True)
        
        current_scores = self.accumulative_marginalization(current_permutation)
        current_auc = self.compute_permutation_auc(current_scores)
        evaluated_permutations.add(tuple(current_permutation))
        
        best_permutation = current_permutation.copy()
        best_auc = current_auc
        reward_baseline = 0.0
        
        with tqdm(total = max_iterations, desc = "Optimizing feature importance", disable = True) as pbar:
            
            iteration = 0
            while iteration < max_iterations:
                i, j = self.probabilistic_sample_swap(current_permutation, tau)
                new_permutation = current_permutation.copy()
                new_permutation[i], new_permutation[j] = new_permutation[j], new_permutation[i]
                
                new_perm_tuple = tuple(new_permutation)
                if new_perm_tuple in evaluated_permutations:
                    pbar.update(1)
                    iteration += 1
                    continue
                
                new_scores = self.accumulative_marginalization(new_permutation)
                new_auc = self.compute_permutation_auc(new_scores)
                evaluated_permutations.add(new_perm_tuple)
                
                reward = current_auc - new_auc
                reward_baseline = baseline_decay * reward_baseline + (1 - baseline_decay) * reward
                advantage = reward - reward_baseline
                
                item_i = current_permutation[i]
                item_j = current_permutation[j]
                
                grad_update = learning_rate * advantage
                self.priors[item_j] += grad_update
                self.priors[item_i] -= grad_update
                
                if reward >= 0:
                    current_permutation = new_permutation
                    current_auc = new_auc
                    if current_auc < best_auc:
                        best_auc = current_auc
                        best_permutation = current_permutation.copy()
                
                pbar.update(1)
                iteration += 1
        
        best_scores = self.accumulative_marginalization(best_permutation)
        best_auc = self.compute_permutation_auc(best_scores)
        
        return best_permutation, best_auc, best_scores