import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from pathlib import Path


# ============================================================
# 1. LOAD RATING FILES
# ============================================================

rating_files = [
    "s234807_ratings.csv",
    "s224202_ratings.csv"
]

participants = {}

for file in rating_files:

    df = pd.read_csv(file)

    # Participant name from filename
    participant = Path(file).stem

    # Mean of the two ratings for each face
    df["MeanRating"] = df[["Rating1", "Rating2"]].mean(axis=1)

    participants[participant] = df

    print(f"\n{participant}")
    print(df.head())


# ============================================================
# 2. COMBINE PARTICIPANTS
# ============================================================

combined = None

for participant, df in participants.items():

    temp = df[["FileName", "MeanRating"]].copy()

    temp = temp.rename(
        columns={"MeanRating": participant}
    )

    if combined is None:
        combined = temp
    else:
        # Match using filename, NOT row position
        combined = combined.merge(
            temp,
            on="FileName",
            how="inner"
        )


# Participant columns
participant_columns = [
    Path(file).stem
    for file in rating_files
]

# Average rating across participants
combined["MeanRating"] = combined[
    participant_columns
].mean(axis=1)


print("\nCombined ratings:")
print(combined.head())

print("\nNumber of faces:")
print(len(combined))

print("\nMean rating statistics:")
print(combined["MeanRating"].describe())

# ============================================================
# REMOVE IMAGES EXCLUDED DURING QUALITY CONTROL
# ============================================================

def image_exists(filename):
    return Path(filename).exists()

n_before = len(combined)

combined = combined[
    combined["FileName"].apply(image_exists)
].copy()

# Reset row numbers after removing images
combined = combined.reset_index(drop=True)

n_after = len(combined)

print(f"\nImages before quality control: {n_before}")
print(f"Images after quality control:  {n_after}")
print(f"Images removed:                {n_before - n_after}")

# ============================================================
# 3. LOAD FACE IMAGES
# ============================================================

images = []

for filename in combined["FileName"]:

    # CSV contains paths such as:
    # FinalData/18_0_0_....jpg
    image_path = Path(filename)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Could not find image: {image_path}"
        )

    # Open image and convert to grayscale
    image = Image.open(image_path).convert("L")

    # Convert image to numpy array
    image = np.asarray(image, dtype=float)

    # Flatten 200 x 200 image into 40,000-dimensional vector
    images.append(image.flatten())


# Stack all faces into one matrix
#
# rows    = faces
# columns = pixels
J = np.stack(images)


print("\nImage matrix shape:")
print(J.shape)


# ============================================================
# 4. CALCULATE THE AVERAGE FACE
# ============================================================

mean_face = J.mean(axis=0)


# ============================================================
# 5. CENTER THE IMAGES
# ============================================================

# Subtract average face from every face.
#
# IMPORTANT:
# Do NOT divide by standard deviation.
# The assignment specifically asks us only to center the images.

J_centered = J - mean_face


print("\nCentered matrix shape:")
print(J_centered.shape)

print("\nMean after centering:")
print(J_centered.mean())


# ============================================================
# 6. DISPLAY THE AVERAGE FACE
# ============================================================

plt.figure(figsize=(5, 5))

plt.imshow(
    mean_face.reshape(200, 200),
    cmap="gray"
)

plt.title("Average Face")
plt.axis("off")

plt.show()


# ============================================================
# 7. PCA
# ============================================================

pca = PCA()

scores = pca.fit_transform(J_centered)


print("\nPCA completed.")

print("PC score matrix shape:")
print(scores.shape)

print("PCA components shape:")
print(pca.components_.shape)


# ============================================================
# 8. EXPLAINED VARIANCE
# ============================================================

import seaborn as sns

explained_variance = pca.explained_variance_ratio_

# Same colors as rating histograms
deep_blue = sns.color_palette("deep")[0]

plt.figure(figsize=(10, 5))

plt.bar(
    np.arange(1, len(explained_variance) + 1),
    explained_variance * 100,
    color=deep_blue,
    edgecolor="black"
)

plt.xlabel("Principal Component")
plt.ylabel("Variance Explained (%)")
plt.title("Variance Explained by Principal Components")

plt.show()


# ============================================================
# 9. CUMULATIVE EXPLAINED VARIANCE
# ============================================================

cumulative_variance = np.cumsum(explained_variance)

for threshold in [0.80, 0.90, 0.95]:

    n_components = (
        np.argmax(cumulative_variance >= threshold) + 1
    )

    print(
        f"{threshold * 100:.0f}% variance: "
        f"{n_components} PCs"
    )


# Plot cumulative variance as well
plt.figure(figsize=(10, 5))

plt.plot(
    np.arange(1, len(cumulative_variance) + 1),
    cumulative_variance * 100
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance Explained (%)")
plt.title("Cumulative Explained Variance")

plt.axhline(80, linestyle="--")
plt.axhline(90, linestyle="--")
plt.axhline(95, linestyle="--")

plt.show()

# ============================================================
# 10. VISUALIZE PRINCIPAL COMPONENTS
# ============================================================

def visualize_pc(pc_number):

    # Python indexing starts at 0
    pc_index = pc_number - 1

    # Weight vector for this principal component
    component = pca.components_[pc_index]

    # Scores of all faces on this component
    pc_scores = scores[:, pc_index]

    # Minimum and maximum observed scores
    min_score = pc_scores.min()
    max_score = pc_scores.max()

    # Reconstruct faces at the two extremes
    min_face = mean_face + min_score * component
    max_face = mean_face + max_score * component

    # Plot:
    # minimum -> average -> maximum
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    axes[0].imshow(
        min_face.reshape(200, 200),
        cmap="gray"
    )
    axes[0].set_title(
        f"Minimum\nscore = {min_score:.1f}"
    )
    axes[0].axis("off")

    axes[1].imshow(
        mean_face.reshape(200, 200),
        cmap="gray"
    )
    axes[1].set_title("Average Face")
    axes[1].axis("off")

    axes[2].imshow(
        max_face.reshape(200, 200),
        cmap="gray"
    )
    axes[2].set_title(
        f"Maximum\nscore = {max_score:.1f}"
    )
    axes[2].axis("off")

    fig.suptitle(f"Principal Component {pc_number}")

    plt.tight_layout()
    plt.show()


# Visualize first six PCs

# for pc in range(1, 7):
    # visualize_pc(pc)

visualize_pc(9)