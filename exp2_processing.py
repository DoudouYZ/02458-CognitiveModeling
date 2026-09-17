import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

files = [
    "s224202_exp2_ratings.csv",
    "s234807_exp2_ratings.csv"
]

for file in files:

    df = pd.read_csv(file)

    rating_cols = [
        "Rating1", "Rating2", "Rating3", "Rating4", "Rating5",
        "Rating6", "Rating7", "Rating8", "Rating9", "Rating10"
    ]

    # Mean rating for each synthetic face
    df["MeanRating"] = df[rating_cols].mean(axis=1)

    # Spearman correlation between predicted and mean observed rating
    rho, p = spearmanr(
        df["PredictedRating"],
        df["MeanRating"]
    )

    participant = file.replace("_exp2_ratings.csv", "")

    print(f"{participant}:")
    print(f"Spearman rho = {rho:.3f}")
    print(f"p = {p:.5f}")

    # Data for boxplot: 10 ratings for each predicted rating
    boxplot_data = [
        df.loc[i, rating_cols].values.astype(float)
        for i in range(len(df))
    ]

    labels = [
        f"{x:.2f}"
        for x in df["PredictedRating"]
    ]

    plt.figure(figsize=(10, 5))

    plt.boxplot(
        boxplot_data,
        tick_labels=labels
    )

    plt.xlabel("Model-predicted rating")
    plt.ylabel("Participant rating")
    plt.title(
        f"{participant} – Experiment 2 "
        f"(Spearman rho = {rho:.3f})"
    )

    plt.ylim(0.5, 5.5)
    plt.tight_layout()

    plt.show()