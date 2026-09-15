import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

# Synthetic faces generated for Experiment 2
images = glob.glob("exp2_data/*.png")

# Sort only so that the final CSV is easier to read.
# Presentation order is still random.
images = sorted(images)

# Number of times each synthetic face should be rated
n_repetitions = 10

# CHANGE THIS for each participant
name = "s224202"


# ============================================================
# CHECK THAT IMAGES WERE FOUND
# ============================================================

if len(images) == 0:
    raise FileNotFoundError(
        "No synthetic images found in exp2_data/"
    )

print(
    f"Found {len(images)} synthetic faces."
)

print(
    f"Each face will be shown {n_repetitions} times."
)

print(
    f"Total trials: {len(images) * n_repetitions}"
)


# ============================================================
# PREPARE DATA STORAGE
# ============================================================

# Count how many times each face has been shown
show_counts = np.zeros(
    len(images),
    dtype=int
)

# rows    = synthetic faces
# columns = repetitions
ratings = np.zeros(
    (len(images), n_repetitions),
    dtype=int
)


# ============================================================
# SET UP IMAGE WINDOW
# ============================================================

plt.ion()

fig, ax = plt.subplots(
    figsize=(6, 6)
)


# ============================================================
# RUN EXPERIMENT
# ============================================================

while np.any(
    show_counts < n_repetitions
):

    # Randomly choose one of the synthetic faces
    image = np.random.choice(
        range(len(images))
    )

    # Only continue if this face still needs
    # to be presented
    if show_counts[image] < n_repetitions:

        image_view = images[image]

        # Display face
        ax.clear()

        ax.imshow(
            plt.imread(image_view),
            cmap="gray"
        )

        ax.axis("off")

        fig.canvas.draw()

        plt.pause(0.001)


        # ====================================================
        # GET PARTICIPANT RATING
        # ====================================================

        try:

            rating = int(
                input(
                    "Please rate the image (1-5): "
                )
            )

        except ValueError:

            print(
                "Invalid input. "
                "Please enter a number between 1 and 5."
            )

            continue


        # Check rating range
        if rating < 1 or rating > 5:

            print(
                "Invalid rating. "
                "Please enter a number between 1 and 5."
            )

            continue


        # ====================================================
        # SAVE THIS RATING
        # ====================================================

        repetition = show_counts[image]

        ratings[
            image,
            repetition
        ] = rating

        show_counts[image] += 1


# ============================================================
# CLOSE IMAGE WINDOW
# ============================================================

plt.close(fig)


# ============================================================
# EXTRACT MODEL-PREDICTED RATINGS FROM FILENAMES
# ============================================================

predicted_ratings = []

for image in images:

    # Example filename:
    #
    # face_01_rating_1.41.png

    filename = Path(image).stem

    predicted_rating = float(
        filename.split("_rating_")[1]
    )

    predicted_ratings.append(
        predicted_rating
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

data = {
    "FileName": images,
    "PredictedRating": predicted_ratings
}


# Add Rating1, Rating2, ... Rating10
for repetition in range(
    n_repetitions
):

    data[
        f"Rating{repetition + 1}"
    ] = ratings[
        :,
        repetition
    ]


df = pd.DataFrame(data)


# ============================================================
# SAVE EXPERIMENT 2 DATA
# ============================================================

output_file = (
    f"{name}_exp2_ratings.csv"
)

df.to_csv(
    output_file,
    index=False
)


print(
    f"\nExperiment complete!"
)

print(
    f"Results saved to: {output_file}"
)

print("\nFirst rows:")

print(
    df.head()
)