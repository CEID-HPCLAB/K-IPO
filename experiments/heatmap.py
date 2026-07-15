import pandas as pd
import seaborn as sns

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import os

plt.rcParams["font.family"] = "DejaVu Serif"; plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 18; plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18; plt.rcParams["legend.fontsize"] = 12
plt.rcParams["legend.title_fontsize"] = 13

plt.rcParams["pdf.fonttype"] = 42; plt.rcParams["ps.fonttype"] = 42

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("custom_blue", ["#e6f0fab2", "#08306b"])

DPI = 1200

DATASETS =  ["abalone", "ai4i2020", "airlines", "bank-customer-churn-prediction", "bank-marketing", "bank32nh", 
             "car-eval-4", "churn", "fried", "japanese-vowels", "lines-overload-50", "magic-gamma-telescope", 
             "mammography", "nhanes", "online-shoppers-purchasing-intention", "pen-digits", "rl", 
             "seismic-bumps", "ur3-cobot-ops", "wilt"]

BASE_DIR = Path(__file__).resolve().parent
pdf_dir = BASE_DIR / "sensitivity_analysis" / "heatmaps" / "pdf"; png_dir = BASE_DIR / "sensitivity_analysis" / "heatmaps" / "png"

os.makedirs(pdf_dir, exist_ok = True); os.makedirs(png_dir, exist_ok = True)

for DATASET in DATASETS:
    
    print(f"[INFO] processing dataset: {DATASET}")

    df = pd.read_csv(f"{BASE_DIR}/sensitivity_analysis/results/{DATASET}.csv")

    tau_ord = df["data_path"].str.extract(r"([0-9.]+)t_([0-9.]+)ord")
    df["tau"] = tau_ord[0].astype(float); df["topk"] = tau_ord[1].astype(float)

    df = df.rename(columns={
        "Accuracy": "accuracy",
        "PR_AUC": "pr_auc",
        "topk_features_overlap": "overlap"
    })

    df["P"] = (df["accuracy"] + df["pr_auc"]) / 2; df["I"] = df["overlap"]
    df["S_harmonic"] = 2 * df["P"] * df["I"] / (df["P"] + df["I"])

    heatmap_data = df.pivot(index = "topk", columns = "tau", values = "S_harmonic")

    heatmap_data = heatmap_data.sort_index(ascending = False)

    fig, ax = plt.subplots(figsize = (6.8, 4.8))

    ax = sns.heatmap(
        heatmap_data,
        annot = True, fmt = ".4f", cmap = cmap,
        linewidths = 4, linecolor = "whitesmoke",
        annot_kws={"fontsize": 18, "fontfamily": "DejaVu Serif"},
        rasterized = False
    )

    cbar = ax.collections[0].colorbar; cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    cbar.set_label("Harmonic Mean ($P$, $I$)", fontsize = 17, labelpad = 13)

    plt.xlabel(r"$\tau$", labelpad = 13, fontsize = 18)
    plt.ylabel("top-k ordering", labelpad = 13, fontsize = 18)

    plt.tight_layout()

    pdf_path = os.path.join(pdf_dir, f"{DATASET}.pdf")
    png_path = os.path.join(png_dir, f"{DATASET}.png")

    plt.savefig(pdf_path, format = "pdf", bbox_inches = "tight")
    plt.savefig(png_path, dpi = DPI, bbox_inches = "tight")

    plt.close()