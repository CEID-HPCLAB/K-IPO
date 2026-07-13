import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from matplotlib.ticker import LogLocator

matplotlib.use("Agg")

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

DATAPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "performance_analysis")

SMOTE_VARIANTS = {"smotewb", "smotenn", "smotenc", "smoten"}

data = {}

for dataset in os.listdir(DATAPATH):
    dataset_path = os.path.join(DATAPATH, dataset)

    if not os.path.isdir(dataset_path):
        continue

    data[dataset] = {}

    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        name = os.path.splitext(file)[0]

        if name.lower() in SMOTE_VARIANTS:
            name = "SMOTE"

        df = pd.read_csv(file_path)
        data[dataset][name] = df

def _format(name: str) -> str:
    return (name.replace("GaussianCopula", "Gaussian Copula"))

order = ["K-IPO", "TabDDPM", "CTGAN", "TVAE", "GaussianCopula", "SMOTE"]

dataset_order = ["ai4i2020", "abalone", "airlines", "seismic-bumps", "bank-customer-churn-prediction", "churn", "bank-marketing", "rl", 
                 "online-shoppers-purchasing-intention", "car-eval-4", "mammography", "ur3-cobot-ops", "nhanes", "magic-gamma-telescope", "fried", 
                 "bank32nh", "wilt", "lines-overload-50", "japanese-vowels", "pen-digits"]

datasets = [d for d in dataset_order if d in data]

labels = {d: f"D{i+1}" for i, d in enumerate(datasets)}
_datasets = [labels[d] for d in datasets]

means = {m: [] for m in order}

for dataset in datasets:
    gens = data[dataset]

    for method in order:
        if method in gens:
            means[method].append(gens[method]["elapsed_time"].mean())
        else:
            means[method].append(np.nan)

x = np.arange(len(datasets)); width = 0.12

fig, ax = plt.subplots(figsize = (14, 5.2))

colors = ["darkorange", "darkviolet", "midnightblue", "darkred", "forestgreen", "darkgoldenrod"]

for i, method in enumerate(order):
    plt.bar(x + i * width, means[method], width, color = colors[i], label = _format(method))
    

plt.xticks(x + width * 2.5, _datasets)
plt.yscale("log") # (log10)

ax.yaxis.set_major_locator(LogLocator(base=10))
ax.minorticks_off()

plt.ylabel("Mean Elapsed Time ($log_{10}$(Seconds))", labelpad = 10)
plt.xlabel("Dataset", labelpad = 10)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, ha = "center")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(True, axis = 'both', linestyle = '--', linewidth = 0.62, alpha = 0.55, zorder = 0) 

ax.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.085), ncol = int(len(colors)), frameon = True, shadow = True,  framealpha = 0.95,   facecolor='white', edgecolor='black')

plt.tight_layout()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pdf_path = os.path.join(BASE_DIR, "external", "pdf", "runtime.pdf")
png_path = os.path.join(BASE_DIR, "external", "png", "runtime.png")

fig.savefig(pdf_path, bbox_inches = "tight")
fig.savefig(png_path, dpi = 1200, bbox_inches = "tight")