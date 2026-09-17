import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rating_files = [
    "s234807_exp3_results.csv",
    "s224202_exp3_results.csv",
    #"test_exp3_results.csv",
]

# Each experiment id maps to a (true value, adaptation level) combination.
true_values = [2.3, 2.5, 2.7]
exp_map = {
    0: (2.3, "Low"),
    1: (2.5, "Low"),
    2: (2.7, "Low"),
    3: (2.3, "High"),
    4: (2.5, "High"),
    5: (2.7, "High"),
}

dataframes = [pd.read_csv(f) for f in rating_files]
df = pd.concat(dataframes, ignore_index=True)
df["TrueValue"] = df["Experiment"].map(lambda e: exp_map[e][0])
df["Adaptation"] = df["Experiment"].map(lambda e: exp_map[e][1])

colors = {"Low": "#2a78d6", "High": "#eb6834"}
bar_width = 0.35
x = np.arange(len(true_values))

fig, ax = plt.subplots(figsize=(9, 6))

for i, adaptation in enumerate(["Low", "High"]):
    means = []
    for tv in true_values:
        subset = df[(df["TrueValue"] == tv) & (df["Adaptation"] == adaptation)]["Rating"]
        means.append(subset.mean())
    offset = (i - 0.5) * bar_width
    bars = ax.bar(
        x + offset,
        means,
        width=bar_width,
        color=colors[adaptation],
        label=f"{adaptation} Adaptation",
        edgecolor="#fcfcfb",
        linewidth=1,
    )

    # Overlay individual data points, jittered within the bar width.
    for j, tv in enumerate(true_values):
        subset = df[(df["TrueValue"] == tv) & (df["Adaptation"] == adaptation)]["Rating"]
        jitter = np.random.uniform(-bar_width / 4, bar_width / 4, size=len(subset))
        ax.scatter(
            np.full(len(subset), x[j] + offset) + jitter,
            subset,
            color="#0b0b0b",
            edgecolor="#fcfcfb",
            linewidth=0.5,
            zorder=3,
            s=30,
        )

ax.set_xlabel("True Value")
ax.set_ylabel("Mean Rating")
ax.set_title("Results of Experiment 3: Rating by True Value and Adaptation Level")
ax.set_xticks(x)
ax.set_xticklabels([str(tv) for tv in true_values])
ax.legend(title="Adaptation")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()
