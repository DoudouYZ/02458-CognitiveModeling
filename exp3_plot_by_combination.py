import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rating_files = [
    "s234807_exp3_results.csv",
    "s224202_exp3_results.csv",
    #"test_exp3_results.csv",
]

# Each experiment id maps to a (true value, adaptation level) combination.
exp_map = {
    0: (2.3, "Low"),
    1: (2.5, "Low"),
    2: (2.7, "Low"),
    3: (2.3, "High"),
    4: (2.5, "High"),
    5: (2.7, "High"),
}
exp_order = [0, 1, 2, 3, 4, 5]
exp_labels = [f"{exp_map[e][0]}\n{exp_map[e][1]} Adaptation" for e in exp_order]
colors = {"Low": "#2a78d6", "High": "#eb6834"}  # dataviz categorical slots 1 & 2

dataframes = [pd.read_csv(f) for f in rating_files]
df = pd.concat(dataframes, ignore_index=True)

x = np.arange(len(exp_order))
means = [df[df["Experiment"] == e]["Rating"].mean() for e in exp_order]
bar_colors = [colors[exp_map[e][1]] for e in exp_order]

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x, means, color=bar_colors, edgecolor="#fcfcfb", linewidth=1)

for j, e in enumerate(exp_order):
    ratings = df[df["Experiment"] == e]["Rating"]
    jitter = np.random.uniform(-0.15, 0.15, size=len(ratings))
    ax.scatter(
        np.full(len(ratings), x[j]) + jitter,
        ratings,
        color="#0b0b0b",
        edgecolor="#fcfcfb",
        linewidth=0.5,
        zorder=3,
        s=30,
    )

ax.set_xlabel("Combination")
ax.set_ylabel("Mean Rating")
ax.set_title("Results of Experiment 3: Rating by Test/Adaptation Combination")
ax.set_xticks(x)
ax.set_xticklabels(exp_labels)

handles = [plt.Rectangle((0, 0), 1, 1, color=colors[a]) for a in ["Low", "High"]]
ax.legend(handles, ["Low Adaptation", "High Adaptation"], title="Adaptation")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()
