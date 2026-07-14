from kipo.importance import compute_importance

import pandas as pd; import numpy as np; import os
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score, 
                             matthews_corrcoef, balanced_accuracy_score, accuracy_score, 
                             precision_score, recall_score, precision_recall_curve, roc_curve)
from sklearn.neighbors import NearestNeighbors

from scikit_posthocs import posthoc_nemenyi_friedman
from scipy.stats import friedmanchisquare, wilcoxon, studentized_range

import matplotlib.pyplot as plt

from pathlib import Path; import yaml

import re; import math

from itertools import combinations


plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 12

plt.rcParams['text.color'] = 'black'; plt.rcParams['axes.labelcolor'] = 'black'; plt.rcParams['xtick.color'] = 'black'; plt.rcParams['ytick.color'] = 'black'


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


def compute_explanations(X_train, y_train, X_test, y_test, estimator, num_cols, cat_cols, mode):
    expl = compute_importance(X_train = X_train, y_train = y_train, estimator = estimator,
                                            X_test = X_test, y_test = y_test, num_cols = num_cols, 
                                            cat_cols = cat_cols, mode = mode)
    if isinstance(expl, pd.Series):
        ranking = expl.sort_values(ascending = False).index.tolist()
    else:
        ranking = expl

    return pd.Series(ranking, name = mode).reset_index(drop = True)


def compute_final_results(results):
    *metrics, _ = results
    
    return pd.Series({k: np.mean([m[k] for m in metrics]) for k in metrics[0]})


def compute_metrics(probs, preds, y_test):
     return {
        'Accuracy': float(accuracy_score(y_test, preds)), 'Precision': float(precision_score(y_test, preds)),
        'F1': float(f1_score(y_test, preds)), 'Recall': float(recall_score(y_test, preds)), 'ROC_AUC': float(roc_auc_score(y_test, probs)),
        'PR_AUC': float(average_precision_score(y_test, probs)), 'MCC': float(matthews_corrcoef(y_test, preds)),
        'BalancedAcc': float(balanced_accuracy_score(y_test, preds))
     }


def separability_N3(X, y):
    X_np = np.array(X); y_np = np.array(y)
    
    nn = NearestNeighbors(); nn.fit(X_np); _, indices = nn.kneighbors(X_np)
    nn_labels = y_np[indices[:, 1]]; n3 = np.mean(nn_labels != y_np)
    
    return 1 - n3


def format_feature_importance(importance, methods):
    rankings = pd.concat(importance, axis = 1); rankings.columns = methods

    return rankings


def compute_importance_overlap(xai_metrics, dataset, topk = 12):
    dataset = dataset if dataset.endswith(".csv") else dataset + ".csv"        

    ground_truth = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "ground_truth", "data", dataset))

    def topk_overlap(col1, col2, k):
        set1 = set(col1.dropna().iloc[:k]); set2 = set(col2.dropna().iloc[:k])
       
        return (len(set1 & set2) / len(set1)) * 100

    gtruth_techniques = ground_truth.columns
    xai_methods = xai_metrics.columns

    overlap_mat = pd.DataFrame(index = gtruth_techniques, columns = xai_methods, dtype = float)

    for g in gtruth_techniques:
        for x in xai_methods:
            overlap_mat.loc[g, x] = topk_overlap(ground_truth[g], xai_metrics[x], k = topk)

    mean_overlap =  overlap_mat.mean().mean()
    
    return mean_overlap 


def compute_topk_ordering(raw_imp, aug_imp):
    raw_feats = raw_imp.sort_values(ascending = False).index
    aug_feats = aug_imp.sort_values(ascending = False).index

    k = 0
    for f_raw, f_aug in zip(raw_feats, aug_feats):
        if f_raw == f_aug:
            k += 1
        else:
            break

    return k


def get_num_topk_features(dataset):
    name = Path(dataset).stem if Path(dataset).suffix == ".csv" else dataset

    config_path = Path(__file__).parent / "topk_features.yml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config.get(name)


def validate_ai4i2020(dataset):
    exp_cols = ["Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Type", "Machine failure"]

    try:
        X, y = dataset

        if set(X.columns) != set(exp_cols[:-1]):
             raise ValueError(f"Invalid input features. Expected {exp_cols[:-1]}, got {list(X.columns)}")

        target_var = y.columns[0] if hasattr(y, "columns") else y.name

        if target_var != exp_cols[-1]:
           raise ValueError(f"Invalid target variable. Expected '{exp_cols[-1]}', got '{getattr(y, 'name', None)}'")

        X.__init__(X[exp_cols[:-1]])
        
        return True

    except Exception as e:
        raise RuntimeError(e)


def compute_hmean(data):
    pred_score = (data["Accuracy"] + data["PR_AUC"]) / 2.0
    denom = pred_score + data["Interpretability"]

    data["PredictiveScore"] = pred_score
    data["HarmonicMean_PI"] = 2.0 * pred_score * data["Interpretability"] / denom

    return data


def aggregate_results(root_dir):
    dirs = [p for p in root_dir.iterdir() if p.is_dir() and p.name != "results"]; dirs = sorted(dirs, key = lambda x: x.name.lower())
    
    rows = []; generators = ["CTGAN", "GaussianCopula", "K-IPO", "TabDDPM", "TVAE", "SMOTEWB"]

    def preprocess(data):
        data.drop(columns = ["data_path", "topk_ordering", "total_samples_before", "total_samples_after"], inplace = True)
        
        data.columns = [re.sub(r'\(\s*(.*?)\s*\)', lambda m: '(' + re.sub(r'\s*-\s*', '-', m.group(1)) + ')', 
                        col.strip()) for col in data.columns]
        
        data.rename(columns = {"topk_features_overlap": "Interpretability"}, inplace = True)

        return data
    
    for data_dir in dirs:
        name = data_dir.name; res_dir = data_dir / "results"

        for gen in generators:
            for candidate in (gen, "SMOTENC", "SMOTEN"):
                path = res_dir / f"{candidate}.csv"
                if path.exists():
                    gen = candidate; break

            res = compute_hmean(preprocess(pd.read_csv(path)))

            mean = res.mean(numeric_only = True); std = res.std(numeric_only = True)

            row = {"Dataset": name, "Method": "SMOTE" if gen in ["SMOTENC", "SMOTEWB", "SMOTEN"] else gen, "File": f"{gen}.csv"}
            row.update(mean.to_dict()); row.update({f"{k}__std": v for k, v in std.items()}); rows.append(row)

    summary = pd.DataFrame(rows)

    metrics = sorted([c for c in summary.select_dtypes(include = np.number).columns if not c.endswith("__std")], key = str.lower)

    summary = summary[["Dataset", "Method", "File"] + [col for metric in metrics for col in (metric, f"{metric}__std")]]

    summary["Method"] = pd.Categorical(summary["Method"], categories = ["CTGAN", "GaussianCopula", "K-IPO", "SMOTE", "TabDDPM", "TVAE"], ordered = True)
    summary = summary.sort_values(["Dataset", "Method"]).reset_index(drop = True)

    return summary


def format_mean_std(mean ,std):
    return (f"{mean:.4f} ± {std:.4f}")


def save_table(data, filepath, index = False, style = "default", decimals = 4):
    filepath = Path(filepath)

    if filepath.suffix.lower() != ".csv":
        filepath = filepath.with_suffix(".csv")
    
    if style == "+/- std":
        metrics = [c for c in data.columns if c not in ["Dataset", "Method", "File"] and not c.endswith("__std")]
        res = data[["Dataset", "Method", "File"]].copy()
        
        for metric in metrics:
            res[metric] = [
                format_mean_std(mean, std) 
                for mean, std in zip(data[metric], data[f"{metric}__std"])
            ]

        res.to_csv(filepath, index=index)
        print(f"[INFO] Saved table: {filepath}")
        return

    data.to_csv(filepath, index = index, float_format = f"%.{decimals}f")

    print(f"[INFO] Saved table: {filepath}")


def compute_winners(data):
    metrics = [c for c in data.columns if c not in ["Dataset", "Method", "File"] and not c.endswith("__std")]
    
    rows = []
    for d_name in sorted(data["Dataset"].unique()):
        
        dataset = data[data["Dataset"] == d_name]
        for metric in metrics:
            rec = dataset[["Method", metric]]; best_val = rec[metric].max(); winners = rec.loc[np.isclose(rec[metric], best_val), "Method"].tolist()

            rows.append({"Dataset": d_name, "Metric": metric,
                "Best_Value": best_val, "Winning_Methods": ", ".join(winners), "Num_Winners": len(winners)})

    return pd.DataFrame(rows)


def compute_method_metric_wins(data):
    metrics = sorted(data["Metric"].unique())
    methods = sorted(data["Winning_Methods"].str.split(", ").explode().unique())

    wins = pd.DataFrame(0, index = methods, columns = metrics, dtype = int)

    for _, row in data.iterrows():
        winners = [w.strip() for w in row["Winning_Methods"].split(",")]
        wins.loc[winners, row["Metric"]] += 1

    wins.index.name = "Method"

    return wins


def build_mat(data, metric):
    methods = data["Method"].unique()

    matrix = (data[["Dataset", "Method", metric]].pivot_table(
            index = "Dataset",
            columns = "Method",
            values = metric,
            aggfunc = "first",
            observed = True,
        ).reindex(columns = methods).dropna(how = "all")
    )

    return matrix


def compute_avg_ranks(mat):
    ranks = mat.rank(axis = 1, method = "average", ascending = False)
    avg_ranks = ranks.mean(axis = 0).sort_values()
    
    return ranks, avg_ranks
    

def statistical_test(data):
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation", "results", "stat_tests")
    Path(os.path.join(BASE_DIR, "tables")).mkdir(parents = True)
    
    metrics = [c for c in data.columns if c not in ["Dataset", "Method", "File"] and not c.endswith("__std")]

    friedman_rows = []; avg_rank_rows = []
    res = {"metric_matrices": {}, "rank_matrices": {}, "avg_ranks": {}, "nemenyi": {}, "wilcoxon": {}, "friedman": {}}

    for metric in metrics:
        print(f"\n[INFO] Processing metric: {metric}")
        metric_ref = re.sub(r"[^\w\-.]+", "_", str(metric).strip()).strip("_")
        metric_mat = build_mat(data, metric)
        
        res["metric_matrices"][metric] = metric_mat
        metric_mat.to_csv(os.path.join(BASE_DIR, "tables", f"{metric_ref}_dataset_method_matrix.csv"), index = True)
        
        rank_mat, avg_ranks = compute_avg_ranks(metric_mat)

        res["rank_matrices"][metric] = rank_mat; res["avg_ranks"][metric] = avg_ranks
        rank_mat.to_csv(os.path.join(BASE_DIR, "tables", f"{metric_ref}_per_dataset_ranks.csv"), index = True)
        avg_ranks.rename("Average_Rank").reset_index().to_csv(os.path.join(BASE_DIR, "tables", f"{metric_ref}_average_ranks.csv"), index = False)

        for method, rank in avg_ranks.items():
            avg_rank_rows.append({"Metric": metric, "Method": method, "Average_Rank": rank})


        friedman = friedman_test(metric_mat); friedman["Metric"] = metric
        friedman_rows.append(friedman); res["friedman"][metric] = friedman
        print(f"[INFO] Friedman test completed (p = {friedman['p_value']:.2e})")

        wilcoxon = pairwise_wilcoxon(metric_mat, metric); res["wilcoxon"][metric] = wilcoxon
        wilcoxon.to_csv(os.path.join(BASE_DIR, "tables", f"{metric_ref}_wilcoxon_pairwise.csv"), index = False)

        print("[INFO] Wilcoxon pairwise tests completed")
        print(f"[INFO] Wilcoxon pairwise results saved: {os.path.join(BASE_DIR, 'tables', f'{metric_ref}_wilcoxon_pairwise.csv')}")

        _, nemenyi = nemenyi_posthoc(metric_mat)
        res["nemenyi"][metric] = nemenyi

        print("[INFO] Nemenyi posthoc test completed")

        if nemenyi is not None:
            nemenyi.to_csv(os.path.join(BASE_DIR, "tables", f"{metric_ref}_nemenyi_pvalues_matrix.csv"), index = True)
            print(f"[INFO] Nemenyi p-value matrix saved: {os.path.join(BASE_DIR, 'tables', f'{metric_ref}_nemenyi_pvalues_matrix.csv')}")

            if avg_ranks is not None and len(avg_ranks) > 0:
                draw_cd_diag(avg_ranks, nemenyi, metric_ref, metric_mat.shape[0])
                print("[INFO] Critical difference diagram saved")


    friedman_rec = pd.DataFrame(friedman_rows)[["Metric", "Num_Datasets", "Num_Methods", "Statistic", "p_value"]]
    save_table(friedman_rec, os.path.join(BASE_DIR, "tables", "friedman_test"), decimals = 20)

    avg_ranks_rec = pd.DataFrame(avg_rank_rows)
    avg_ranks_rec.to_csv(os.path.join(BASE_DIR, "tables", "average_ranks.csv"))
    avg_ranks_table = (avg_ranks_rec.pivot(index = "Method", columns = "Metric", values = "Average_Rank").
                                    reindex(["CTGAN", "GaussianCopula", "K-IPO", "SMOTE", "TabDDPM", "TVAE"]))

    avg_ranks_table.to_csv(os.path.join(BASE_DIR, "tables", "average_ranks_table.csv"), index = True)

    nemenyi_table = create_nemenyi_table(res["nemenyi"])
    save_table(nemenyi_table, os.path.join(BASE_DIR, "tables", "nemenyi_posthoc"))

    nemenyi_sign_table = nemenyi_table[nemenyi_table["Significant_at_0.05"] == "Yes"].copy()
    save_table(nemenyi_sign_table, os.path.join(BASE_DIR, "tables", "nemenyi_posthoc_significant"))
    print("\n[INFO] Statistical analysis completed")


def friedman_test(mat):
    comp = mat.dropna()
    n_datasets, n_methods = comp.shape

    stat, p = friedmanchisquare(*(comp[c].values for c in comp.columns))

    return {"Num_Datasets": n_datasets, "Num_Methods": n_methods, "Statistic": stat, "p_value": p}


def pairwise_wilcoxon(mat, metric):
    rows = []

    for m1, m2 in combinations(mat.columns, 2):
        pair = mat[[m1, m2]].dropna(); n = len(pair); base = {"Metric": metric, "Method_1": m1, "Method_2": m2, "Num_Datasets": n}

        diffs = pair[m1] - pair[m2]; zeros = int(np.isclose(diffs, 0).sum())

        if np.allclose(diffs, 0):
            rows.append({**base, "Num_Zero_Diffs": zeros, "Statistic": 0.0, "p_value": 1.0,
                         "Usable": True, "Reason": "All paired differences are zero."})
            continue

        try:
            stat, p = wilcoxon(pair[m1], pair[m2], zero_method = "wilcox",
                               alternative = "two-sided", method = "approx")
            
            rows.append({**base, "Num_Zero_Diffs": zeros, "Statistic": stat, "p_value": p, "Usable": True, "Reason": ""})

        except Exception as e:
            rows.append({**base, "Num_Zero_Diffs": zeros, "Statistic": np.nan, "p_value": np.nan, "Usable": False, "Reason": str(e)})

    return pd.DataFrame(rows)


def nemenyi_posthoc(mat):
    comp = mat.dropna(axis = 0, how = "any")
    
    return comp, posthoc_nemenyi_friedman(comp).loc[comp.columns, comp.columns]


def draw_cd_diag(avg_ranks, nemenyi_pmat, metric, n_datasets):
    avg_ranks = avg_ranks.sort_values(); methods = list(avg_ranks.index)
    xvals = avg_ranks.values.astype(float); n_methods = len(methods)

    step = 10; min_rank = math.floor(np.min(xvals) * step) / step; max_rank = math.ceil(np.max(xvals) * step) / step

    fig, ax = plt.subplots(figsize = (5.2, 2.2))

    X_MARGIN = 0.25
    ax.set_xlim(min_rank - X_MARGIN, max_rank + X_MARGIN); ax.axis("off")

    side_map = set_label_sides(xvals)

    left_elems = sorted([i for i in range(n_methods) if side_map[i] == "left"], key = lambda i: xvals[i])
    right_elems = sorted([i for i in range(n_methods) if side_map[i] == "right"], key = lambda i: xvals[i], reverse = True)

    BASE = -0.33; STEP = 0.06
    left_y = [BASE - i * STEP for i in range(len(left_elems))]; right_y = [BASE - i * STEP for i in range(len(right_elems))]

    label_specs = []
    for elems, ys, side in [(left_elems, left_y, "left"), (right_elems, right_y, "right")]:
        for k, idx in enumerate(elems):
            method = methods[idx]; x = xvals[idx]; y = ys[k]
            
            label = f"{'Gaussian Copula' if method == 'GaussianCopula' else method} ({x:.2f})"
            label_specs.append({"idx": idx, "method": method, "x": x, "y": y, "side": side, "label": label})

    FONTSIZE = 8
   
    left_extra = right_extra = 0
    for spec in label_specs:
        width_est = max(0.05, len(spec["label"]) * 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]) * FONTSIZE / 8)
        
        if spec["side"] == "left":
            left_extra = max(left_extra, width_est + 0.1)
        else:
            right_extra = max(right_extra, width_est + 0.1)

    ax.set_xlim(min_rank - left_extra - 0.1, max_rank + right_extra + 0.1)
    ax.plot([min_rank - 0.1, max_rank + 0.1], [0, 0], color = "black", lw = 1.0)

    tick_step = 0.25 if max_rank - min_rank <= 2.5 else 0.5
    tick_start = math.floor(min_rank / tick_step) * tick_step
    tick_end = math.ceil(max_rank / tick_step) * tick_step
    
    ticks = np.arange(tick_start, tick_end + tick_step / 10, tick_step)

    for t in ticks:
        if min_rank - 0.12 <= t <= max_rank + 0.12:
            ax.plot([t, t], [0, 0 + 0.045], color = "black", lw = 0.8)
            tick_label = f"{t:.2f}".rstrip("0").rstrip(".")
            ax.text(t, 0 + 0.06, tick_label, ha = "center", va = "bottom", fontsize = FONTSIZE)

    groups = get_nonsign_groups(avg_ranks, nemenyi_pmat, alpha = 0.05)

    used_levels = []; start, gap = -0.21, 0.10

    for i, j in groups:
        x1, x2 = avg_ranks.iloc[i], avg_ranks.iloc[j]

        level = 0
        while any(lev == level and not (x2 < gx1 or x1 > gx2) for lev, gx1, gx2 in used_levels):
            level += 1

        y = start - level * gap; used_levels.append((level, x1, x2))
        
        ax.plot([x1, x2], [y, y], color = "black", lw = 1.3, solid_capstyle = "butt")

    stub = 0.02

    for x, y, label, side, method in [(s["x"], s["y"], s["label"], s["side"], s["method"]) for s in label_specs]:
        color = "#08306b" if method == "K-IPO" else "black"
        
        ax.scatter(x, 0, s = 34, color = color, zorder = 3)
        ax.plot([x, x], [0, y], color = color, lw = 0.9)

        dx = -stub if side == "left" else stub; ha = "right" if side == "left" else "left"

        ax.plot([x, x + dx], [y, y], color = "black" , lw = 0.9)
        ax.text(x + dx + (-0.02 if side == "left" else 0.02), y, label, ha = ha, va = "center", fontsize = FONTSIZE, color = color)

    fig.tight_layout(pad = 0.1)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    pdf_dir = os.path.join(BASE_DIR, "evaluation", "results", "stat_tests", "critical_diagrams", "pdf")
    png_dir = os.path.join(BASE_DIR, "evaluation", "results", "stat_tests", "critical_diagrams", "png")

    os.makedirs(pdf_dir, exist_ok = True)
    os.makedirs(png_dir, exist_ok = True)

    pdf_path = os.path.join(pdf_dir, f"{metric.lower()}.pdf")
    png_path = os.path.join(png_dir, f"{metric.lower()}.png")

    fig.savefig(pdf_path, bbox_inches = "tight", facecolor = "white", edgecolor = "none")
    fig.savefig(png_path, dpi = 1200, bbox_inches = "tight", facecolor = "white", edgecolor = "none")

    plt.close(fig)


def set_label_sides(pos):
    n = len(pos); c = [0, 0]
    
    sides = {}

    for r, i in enumerate(np.argsort(pos)):
        s = 0 if r < n/3 else 1 if r >= 2*n/3 else int(c[1] < c[0])
        sides[i] = "left" if s == 0 else "right"
        c[s] += 1

    return sides


def get_nonsign_groups(avg_ranks, nemenyi_pmat, alpha):
    methods = list(avg_ranks.index)
    groups = []

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            if all(nemenyi_pmat.loc[a, b] >= alpha for a, b in combinations(methods[i:j+1], 2)):
                groups.append((i, j))
            else:
                break

    return [g for g in groups if not any(h != g and h[0] <= g[0] and h[1] >= g[1] for h in groups)]


def create_nemenyi_table(mats):
    rows = []

    for metric, pmat in mats.items():
        if pmat is None or pmat.empty:
            continue

        for m1, m2 in combinations(pmat.index, 2):
            pval = pmat.loc[m1, m2]
            rows.append({"Metric": metric, "Method_1": m1, "Method_2": m2,
                "Nemenyi_p_value": pval, f"Significant_at_0.05": "Yes" if pd.notna(pval) and pval < 0.05 else "No"
            })

    return pd.DataFrame(rows).sort_values(["Metric", "Nemenyi_p_value", "Method_1", "Method_2"]).reset_index(drop = True) if rows else pd.DataFrame()