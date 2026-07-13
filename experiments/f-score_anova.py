import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os 

raw_scores = np.array([45.0, 25.0, 13.0, 8.0, 2.5, 2.0, 1.6, 1.3, 1.0, 0.6])

scores = raw_scores / raw_scores.sum(); cum = np.cumsum(scores)

threshold = 0.90; k = int(np.argmax(cum >= threshold) + 1)

x = np.arange(1, len(scores) + 1)

plt.rcParams["font.family"] = "DejaVu Serif"; plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 18; plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18; plt.rcParams["legend.fontsize"] = 12

plt.rcParams["pdf.fonttype"] = 42; plt.rcParams["ps.fonttype"] = 42

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("custom_blue", ["#08306b", "#e6f0fa",])

colors = [cmap(i / (len(scores) - 1)) for i in range(len(scores))]

fig, ax = plt.subplots(figsize = (9.5, 5.2))

ax.add_patch(Rectangle((0.5, 0), k, scores.max() * 1.15, facecolor = "lightblue", alpha = 0.19, edgecolor = "none", zorder = 0))

ax.bar(x, scores, width = 0.85, color = colors, edgecolor = "none", zorder = 3)
ax.set_xlim(0.5, len(scores) + 0.4); ax.set_ylim(0, scores.max() * 1.05)

ax.set_xlabel("Features sorted by ANOVA F-score", fontsize = 18, labelpad = 13)
ax.set_ylabel("ANOVA F-score", fontsize = 18, labelpad = 13)

ax.set_xticks(x); ax.set_xticklabels([f"$f_{{{i}}}$" for i in x])
ax.grid(axis = "y", alpha = 0.12, linewidth = 0.6); ax.set_axisbelow(True)

ax.set_ylabel("ANOVA F-score", fontsize = 18); ax.tick_params(axis = 'y')

ax2 = ax.twinx()
ax2.plot(x, cum, color = "#556B2F", marker = "o", linewidth = 2, zorder = 4)

x_offsets = [0.00, 0.43, 0.46, 0.08, 0.08, 0.078, 0.078, 0.078, 0.078, 0.02]
y_offsets = [-0.075, -0.06, -0.055, 0.02, 0.02, 0.02, 0.018, 0.018, 0.018, 0.0185]

for i, (xi, yi) in enumerate(zip(x, cum)):
    ax2.text(xi + x_offsets[i], yi + y_offsets[i], f"{yi*100:.1f}%", color = "#556B2F", 
             ha = "center", va = "bottom", fontsize = 13)

ax2.set_ylim(0, 1.02)
ax2.set_ylabel("Cumulative Contribution", rotation = 270, color = "#556B2F", fontsize = 18, labelpad = 13)

ax2.spines["right"].set_color( "#556B2F")

right_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
ax2.set_yticks(right_ticks)
ax2.set_yticklabels([f"{int(t*100)}%" for t in right_ticks])

ax2.tick_params(axis = 'y', colors = '#556B2F')

for tick, val in zip(ax2.get_yticklabels(), right_ticks):
    if np.isclose(val, 0.9):
        tick.set_color("#E0541D"); tick.set_fontweight("bold")


ax2.axhline(threshold, color = "#E0541D", linestyle = (0, (4, 3)), linewidth = 1.4, zorder = 2)
ax2.axvline(k + 0.5, color = "#E0541D", linestyle = (0, (4, 3)), linewidth = 1.4, zorder = 2)

ax.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False); ax2.spines["left"].set_visible(False)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pdf_path = os.path.join(BASE_DIR, "external", "pdf", "f-score_anova.pdf")
png_path = os.path.join(BASE_DIR, "external", "png", "f-score_anova.png")

fig.savefig(pdf_path, bbox_inches = "tight")
fig.savefig(png_path, dpi = 1200, bbox_inches = "tight")

plt.close(fig)