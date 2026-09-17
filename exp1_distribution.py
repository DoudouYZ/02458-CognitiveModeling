import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# -----------------------------
# 1. Load Experiment 1 ratings
# -----------------------------

deep_colors = sns.color_palette("deep", 5)

rating_files = [
    "s234807_ratings.csv",
    "s224202_ratings.csv"
]

participants = {}

for file in rating_files:
    df = pd.read_csv(file)

    # Keep only images that still exist in the FinalData folder
    df = df[
        df["FileName"].apply(lambda filename: Path(filename).exists())
    ].copy().reset_index(drop=True)

    participant = Path(file).stem
    participants[participant] = df

    print(f"\n{participant}")
    print(df.head())
    print("Rating 1 range:", df["Rating1"].min(), "-", df["Rating1"].max())
    print("Rating 2 range:", df["Rating2"].min(), "-", df["Rating2"].max())

    # Combine the two presentations for the histogram
    ratings = pd.concat([df["Rating1"], df["Rating2"]])

    plt.figure()

    counts, bins, patches = plt.hist(
        ratings,
        bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        edgecolor="black"
    )

    # Give each rating bar a different color
    for patch, color in zip(patches, deep_colors):
        patch.set_facecolor(color)

    plt.xticks([1, 2, 3, 4, 5])
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.title(f"Ratings — {participant}")

    plt.show()


for participant, df in participants.items():
    df["MeanRating"] = df[["Rating1", "Rating2"]].mean(axis=1)

    combined = None

for participant, df in participants.items():

    temp = df[["FileName", "MeanRating"]].copy()
    temp = temp.rename(columns={"MeanRating": participant})

    if combined is None:
        combined = temp
    else:
        combined = combined.merge(
            temp,
            on="FileName",
            how="inner"
        )

print(combined.head())

participant_columns = rating_files.copy()
participant_columns = [Path(f).stem for f in participant_columns]

combined["MeanRating"] = combined[participant_columns].mean(axis=1)

print(combined.head())
print(combined["MeanRating"].describe())