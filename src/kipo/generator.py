import pandas as pd; import numpy as np
from time import perf_counter as time

from imblearn.over_sampling import SMOTENC, SMOTEN, RandomOverSampler
from imblearn.combine import SMOTEENN

from smote_variants import SMOTEWB

from ctgan import CTGAN, TVAE

from sdv.single_table import GaussianCopulaSynthesizer


def generate_data_pool(generator, X_train, y_train, chunk_size, kipo = True, **gen_args):

    match generator.lower():
        
        case "smotenc" | "smoteenn" | "smoten" | "ros":
            req_samples = int((y_train.squeeze() == 1).sum()) + chunk_size

            if "random_state" not in gen_args:
                gen_args["random_state"] = np.random.default_rng(int(time())).integers(0, 2**31 - 1).item()
            
            gen_args["sampling_strategy"] = {
                1: req_samples,
                0: int((y_train.squeeze() == 0).sum())
            }
            
            if generator.lower() == "smotenc":
                X_aug, y_aug = SMOTENC(**gen_args).fit_resample(X_train, y_train)
            
            elif generator.lower() == "smoten":
                X_aug, y_aug = SMOTEN(**gen_args).fit_resample(X_train, y_train)
            
            elif generator.lower() == "ros":
                X_aug, y_aug = RandomOverSampler(**gen_args).fit_resample(X_train, y_train)
            
            else:
                X_aug, y_aug = SMOTEENN(**gen_args).fit_resample(X_train, y_train)

            return (
                X_aug.iloc[- len(y_aug) + len(y_train):]
                if kipo else (X_aug, y_aug)
            )
        
        case "smotewb":
            diff = int((y_train.squeeze() == 0).sum()) - int((y_train.squeeze() == 1).sum()) 
            gen_args["proportion"] = chunk_size / diff

            if "random_state" not in gen_args:
                gen_args["random_state"] = np.random.default_rng(int(time())).integers(0, 2**31 - 1).item()
            
            sampler = SMOTEWB(**gen_args); X_aug, y_aug = sampler.sample(X_train.values, y_train.squeeze().values)
            X_aug = pd.DataFrame(X_aug, columns = X_train.columns); y_aug = pd.DataFrame(y_aug, columns = y_train.columns)

            return (
                X_aug.iloc[- len(y_aug) + len(y_train):]
                if kipo else (X_aug, y_aug)
            )
        
        case "gaussiancopula":
            _ = gen_args.pop("random_state", "")

            (gen := GaussianCopulaSynthesizer(**gen_args)).fit(X_train[y_train[y_train.columns[0]] == 1])
            
            gen_samples = gen.sample(chunk_size); X_aug = pd.concat([X_train, gen_samples], ignore_index = True)
            y_aug = pd.concat([y_train, pd.DataFrame({y_train.columns[0]: [1] * chunk_size})], ignore_index = True)
        
            return (X_aug, y_aug) if not kipo else X_aug.iloc[X_train.shape[0]: ]
        
        case "ctgan" | "tvae":
            disc_cols = gen_args.pop("discrete_features", []); _ = gen_args.pop("random_state", "")

            if generator.lower() == "ctgan":
                (gen := CTGAN(**gen_args)).fit(X_train[y_train[y_train.columns[0]] == 1],
                    discrete_columns = disc_cols)
    
            else:
                (gen := TVAE(**gen_args)).fit(X_train[y_train[y_train.columns[0]] == 1],
                    discrete_columns = disc_cols)
                
            gen_samples = gen.sample(chunk_size); X_aug = pd.concat([X_train, gen_samples], ignore_index = True)
            
            y_aug = pd.concat([y_train, pd.DataFrame({y_train.columns[0]: [1] * chunk_size})], ignore_index = True)
        
            return (X_aug, y_aug) if not kipo else X_aug.iloc[X_train.shape[0]: ]

        case _:
            raise ValueError(f"[ERROR] Unknown generator: {generator}. Supported generators are: smotenc, smoteenn, smoten, ros, smotewb, gaussiancopula, ctgan, tvae.")