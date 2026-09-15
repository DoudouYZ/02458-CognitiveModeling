import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import KFold


# ============================================================
# 1. LOAD RATING FILES FROM EXPERIMENT 1
# ============================================================

rating_files = [
    "s234807_ratings.csv",
    "s224202_ratings.csv"
]

participants = {}

for file in rating_files:

    df = pd.read_csv(file)

    participant = Path(file).stem

    # Average the two repeated ratings
    df["MeanRating"] = df[
        ["Rating1", "Rating2"]
    ].mean(axis=1)

    participants[participant] = df


# ============================================================
# 2. COMBINE PARTICIPANTS
# ============================================================

combined = None

for participant, df in participants.items():

    temp = df[
        ["FileName", "MeanRating"]
    ].copy()

    temp = temp.rename(
        columns={"MeanRating": participant}
    )

    if combined is None:
        combined = temp

    else:
        combined = combined.merge(
            temp,
            on="FileName",
            how="inner"
        )


participant_columns = [
    Path(file).stem
    for file in rating_files
]

# Average ratings across participants
combined["MeanRating"] = combined[
    participant_columns
].mean(axis=1)


# ============================================================
# 3. REMOVE FACES EXCLUDED DURING QUALITY CONTROL
# ============================================================

combined = combined[
    combined["FileName"].apply(
        lambda filename: Path(filename).exists()
    )
].copy()

combined = combined.reset_index(drop=True)

print(
    f"Number of faces after quality control: "
    f"{len(combined)}"
)


# ============================================================
# 4. LOAD IMAGES
# ============================================================

images = []

for filename in combined["FileName"]:

    image = Image.open(
        filename
    ).convert("L")

    image = np.asarray(
        image,
        dtype=float
    )

    images.append(
        image.flatten()
    )


J = np.stack(images)

print(
    f"Image matrix: {J.shape}"
)


# ============================================================
# 5. CENTER IMAGES
# ============================================================

mean_face = J.mean(axis=0)

J_centered = J - mean_face


# ============================================================
# 6. PCA
# ============================================================

pca = PCA()

scores = pca.fit_transform(
    J_centered
)


# ============================================================
# 7. KEEP PCs EXPLAINING 90% OF VARIANCE
# ============================================================

cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

n_candidate_pcs = (
    np.argmax(
        cumulative_variance >= 0.90
    )
    + 1
)

X = scores[
    :,
    :n_candidate_pcs
]

y = combined[
    "MeanRating"
].to_numpy()

print(
    f"PCs needed for 90% variance: "
    f"{n_candidate_pcs}"
)


# ============================================================
# 8. FORWARD SELECTION
# ============================================================

model = LinearRegression()

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

selector = SequentialFeatureSelector(
    model,
    direction="forward",
    n_features_to_select="auto",
    tol=0.001,
    scoring="neg_mean_squared_error",
    cv=cv,
    n_jobs=-1
)

selector.fit(
    X,
    y
)


selected_mask = selector.get_support()

selected_indices = np.where(
    selected_mask
)[0]

selected_pcs = (
    selected_indices + 1
)

print(
    f"Selected PCs: {selected_pcs}"
)


# ============================================================
# 9. FIT FINAL REGRESSION MODEL
# ============================================================

X_selected = X[
    :,
    selected_indices
]

final_model = LinearRegression()

final_model.fit(
    X_selected,
    y
)

print(
    f"Intercept: "
    f"{final_model.intercept_:.4f}"
)


# ============================================================
# 10. CHECK PREDICTED TRAINING RANGE
# ============================================================

predicted_ratings = final_model.predict(
    X_selected
)

min_predicted = predicted_ratings.min()
max_predicted = predicted_ratings.max()

print(
    f"Predicted training range: "
    f"{min_predicted:.2f} to "
    f"{max_predicted:.2f}"
)


# ============================================================
# 11. EXPERIMENT 3 TARGET RATINGS
# ============================================================

# Three faces around 2.5
# One extreme low face
# One extreme high face

target_ratings = [
    1.0,
    2.3,
    2.5,
    2.7,
    5.0
]

print("\nExperiment 3 target ratings:")

for rating in target_ratings:
    print(rating)


# ============================================================
# 12. PREPARE REGRESSION PARAMETERS
# ============================================================

w = final_model.coef_

delta = final_model.intercept_

w_norm_squared = np.sum(
    w ** 2
)

print(
    f"\n||w||^2 = "
    f"{w_norm_squared:.10f}"
)


# ============================================================
# 13. GENERATE EXPERIMENT 3 FACES
# ============================================================

synthetic_faces = []


for target_rating in target_ratings:

    # Equation 2.27
    alpha = (
        target_rating - delta
    ) / w_norm_squared


    # Equation 2.28
    synthetic_selected_scores = (
        alpha * w
    )


    # Sanity check:
    # Does the regression model predict the desired value?
    check_rating = (
        synthetic_selected_scores @ w
        + delta
    )

    print(
        f"Target = {target_rating:.1f}, "
        f"model prediction = {check_rating:.3f}, "
        f"alpha = {alpha:.2f}"
    )


    # Create full PCA score vector
    synthetic_pca_scores = np.zeros(
        pca.n_components_
    )

    synthetic_pca_scores[
        selected_indices
    ] = synthetic_selected_scores


    # Convert PCA representation back into pixels
    synthetic_centered_face = (
        synthetic_pca_scores
        @ pca.components_
    )


    # Add the average face back
    synthetic_face = (
        mean_face
        + synthetic_centered_face
    )


    synthetic_faces.append(
        synthetic_face
    )


synthetic_faces = np.stack(
    synthetic_faces
)


# ============================================================
# 14. CREATE EXPERIMENT 3 FOLDER
# ============================================================

output_folder = Path(
    "exp3_data"
)

output_folder.mkdir(
    exist_ok=True
)


# ============================================================
# 15. SAVE EXPERIMENT 3 FACES
# ============================================================

print("\nSaving Experiment 3 faces:")


for face, rating in zip(
    synthetic_faces,
    target_ratings
):

    # Restrict pixel values to valid image range
    face_uint8 = np.clip(
        face,
        0,
        255
    ).astype(
        np.uint8
    )


    # Convert flattened 40,000 pixels
    # back into 200 x 200 image
    face_image = Image.fromarray(
        face_uint8.reshape(200, 200)
    )


    filename = (
        output_folder
        / f"face_rating_{rating:.1f}.png"
    )


    face_image.save(
        filename
    )


    print(
        f"Saved: {filename}"
    )


# ============================================================
# 16. FINISHED
# ============================================================

print(
    f"\nFinished! "
    f"{len(synthetic_faces)} faces "
    f"saved in '{output_folder}'."
)