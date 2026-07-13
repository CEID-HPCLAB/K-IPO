import pandas as pd; import numpy as np
from copy import deepcopy
from time import perf_counter as time

from kipo.importance import compute_importance, kendall_tau, topk_ordering, topk_mag, topk_overlap
from kipo.generator import generate_data_pool


class KIPOSelector:
    
    def __init__(self, num_features: int, tau_threshold: float = 0.7, feature_importance = None,
                 topk_ordering: int = None, topk_overlap: int = None, magnitude_threshold: float = None, max_block_size = None, 
                 min_block_size = None, max_dry_attempts: int = 25,**imp_args):
        
        self.num_features = num_features; self.feature_importance = feature_importance; 
        self.max_block_size = max_block_size; self.min_block_size = min_block_size; self.magnitude_threshold = magnitude_threshold
        self.imp_args = imp_args; self.max_dry_attempts = max_dry_attempts

        if not (0 < tau_threshold <= 1):
            raise ValueError("[ERROR] Invalid value for parameter tau_threshold. Expected a float in (0, 1]")
        
        self.tau_threshold = tau_threshold

        self.topk_ordering = self._parse_topk(topk_ordering, "topk_ordering"); self.topk_overlap = self._parse_topk(topk_overlap, "topk_overlap")

    
    def _accept(self, chunk, X_train, y_train):
        aug_imp = compute_importance(mode = "f-score_ANOVA", X_train = pd.concat([X_train, chunk], ignore_index = True),
                                    y_train = pd.concat([y_train, pd.DataFrame({y_train.columns[0]: [1] * chunk.shape[0]})], ignore_index = True).to_numpy().ravel(), 
                                    **self.imp_args)
        
        if kendall_tau(self.feature_importance, aug_imp) < self.tau_threshold:
            return False
        
        if self.topk_overlap is not None and not topk_overlap(self.feature_importance, aug_imp, self.topk_overlap):
            return False
        
        if self.topk_ordering is not None and not topk_ordering(self.feature_importance, aug_imp, self.topk_ordering):
            return False

        if (self.topk_ordering is not None and self.magnitude_threshold is not None and
            (np.any(topk_mag(self.feature_importance, aug_imp, self.topk_ordering) > self.magnitude_threshold) or np.any(topk_mag(self.feature_importance, aug_imp, self.topk_ordering) < 2 - self.magnitude_threshold))
        ):  
            return False
        
        return True
        
    
    def select(self, X_train, y_train, X_test, y_test, ratio = 1, generator = "smotenc", preprocessing = None, **gen_args):
        if self.feature_importance is None:
            self.feature_importance = compute_importance(mode = "f-score_ANOVA",
                                                        X_train = pd.concat([X_train, X_test], axis = 0), 
                                                        y_train = pd.concat([y_train, y_test], axis = 0).values.ravel(), 
                                                        X_test = X_test, y_test = y_test.values.ravel(), 
                                                        **self.imp_args) 
                                                           
        major = int((y_train == 0).sum().iloc[0]); minor = int((y_train == 1).sum().iloc[0])
        total_needed_samples = max(0, int(major * ratio) - minor)
        
        self.max_block_size = min(max(total_needed_samples // 4, 2500) if self.max_block_size is None else self.max_block_size, total_needed_samples)
        self.min_block_size =  min(int(total_needed_samples * 0.05), 50) if self.min_block_size is None else self.min_block_size

        seed_generator = np.random.default_rng(int(time())); gen_args.setdefault("random_state", seed_generator.integers(0, 2**31 - 1).item())

        print(f"[INFO] total number of requested samples: {total_needed_samples}, generator: {generator}, importance mode: f-score_ANOVA, max block size: {self.max_block_size}, min block size: {self.min_block_size}")

        extra = [f"kendall's tau threshold: {self.tau_threshold}"]

        if getattr(self, "topk_overlap", None) is not None:
            extra.append(f"topk overlap: {self.topk_overlap}")
        
        if getattr(self, "topk_ordering", None) is not None:
            extra.append(f"topk ordering: {self.topk_ordering}")

        print("[INFO]", ", ".join(extra), flush = True)

        while total_needed_samples > 0:
            current_seed = seed_generator.integers(0, 2**31 - 1); gen_args["random_state"] = int(current_seed)
            
            block_size = min(self.max_block_size, total_needed_samples); self.min_block_size = min(self.min_block_size, block_size)

            data_pool = generate_data_pool(generator, X_train, y_train, block_size, **gen_args).reset_index(drop = True)

            if self._accept(data_pool, X_train, y_train):
                total_needed_samples -= data_pool.shape[0]; X_train = pd.concat([X_train, data_pool], ignore_index = True)
                y_train = pd.concat([y_train, pd.DataFrame({y_train.columns[0]: [1] * data_pool.shape[0]})], ignore_index = True)
            
                continue
        
            i = 1;  dry_attempts = 0
            
            while (data_pool.shape[0] // 2**i >= self.min_block_size):
            
                chunk_size = data_pool.shape[0] // 2**i
                drop_idxs = []; accepted_any = False

                for start_idx in range(0, data_pool.shape[0], chunk_size):
                    end_idx = min(start_idx + chunk_size, data_pool.shape[0]); chunk = data_pool.iloc[start_idx: end_idx]

                    if chunk.shape[0] < self.min_block_size:
                        break
                    
                    if self._accept(chunk, X_train, y_train):
                        total_needed_samples -= chunk.shape[0]; X_train = pd.concat([X_train, chunk], ignore_index = True)
                        y_train = pd.concat([y_train, pd.DataFrame({y_train.columns[0]: [1] * chunk.shape[0]})], ignore_index = True)
                        
                        drop_idxs.extend(range(start_idx, end_idx)); accepted_any = True; dry_attempts = 0
                    
                    else:
                        dry_attempts += 1
                        if dry_attempts >= self.max_dry_attempts:
                            break
                
                if dry_attempts >= self.max_dry_attempts:
                    break  
                            
                if not accepted_any:
                    pass
                
                else:
                    data_pool = data_pool.drop(drop_idxs); data_pool.reset_index(drop = True, inplace = True)
                
                i += 1
                    
        aug_imp = compute_importance(mode = "f-score_ANOVA", X_train = X_train, y_train = y_train.values.ravel(), 
                                    X_test = X_test, y_test = y_test.values.ravel(), preprocessing = preprocessing, **self.imp_args)   
        
        info = {
            "generator": generator,
            "requested_samples": int(major * ratio) - minor,
            "kendall_tau_threshold": self.tau_threshold,
            "achieved_kendall_tau": kendall_tau(self.feature_importance, aug_imp),
            "topk_ordering": self.topk_ordering,
            "topk_overlap": self.topk_overlap,
            "minority_class_samples_before": minor,
            "minority_class_samples_after": int((y_train == 1).sum().iloc[0]),
            "majority_class_samples": major,
            "max_block_size": self.max_block_size,
            "min_block_size": self.min_block_size,
        }

        return X_train, y_train, info
    

    def _parse_topk(self, value, name):
        if value is None or value == 0:
            return None

        if isinstance(value, float) and 0 <= value <= 1:
            k = round(value * self.num_features)
            
            if k >= 1:
                return k

        if isinstance(value, (int, float)) and value >= 1 and float(value).is_integer():
            k = int(value)
            
            if k <= self.num_features:
                return k
                
        raise ValueError(f"[ERROR] Invalid value for parameter {name}. Expected an integer in [1, {self.num_features}] or a float in [0, 1]")