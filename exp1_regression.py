import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import KFold, cross_val_score


# ============================================================
# 1. LOAD RATINGS
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

# Average rating across participants
combined["MeanRating"] = combined[
    participant_columns
].mean(axis=1)


# ============================================================
# 3. REMOVE IMAGES EXCLUDED DURING QUALITY CONTROL
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

    image = Image.open(filename).convert("L")

    image = np.asarray(
        image,
        dtype=float
    )

    images.append(
        image.flatten()
    )


J = np.stack(images)

print("Image matrix:", J.shape)


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

print("PCA score matrix:", scores.shape)


# ============================================================
# 7. SELECT PCA SUBSPACE
# ============================================================

# Start with PCs explaining 90% of image variance.
#
# This does NOT mean all these PCs will enter the regression.
# It simply defines the candidate PC subspace from which
# forward selection can choose.

cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

n_candidate_pcs = (
    np.argmax(cumulative_variance >= 0.90) + 1
)

print(
    f"PCs needed for 90% variance: "
    f"{n_candidate_pcs}"
)

X = scores[:, :n_candidate_pcs]

y = combined["MeanRating"].to_numpy()


print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 8. FORWARD SELECTION
# ============================================================

model = LinearRegression()

# Same folds are used throughout selection
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

selector = SequentialFeatureSelector(
    model,

    # Forward selection
    direction="forward",

    # Let sklearn stop when adding another PC
    # no longer sufficiently improves CV performance
    n_features_to_select="auto",

    # Require an improvement to continue
    tol=0.001,

    # Evaluate candidate models using cross-validation
    scoring="neg_mean_squared_error",

    cv=cv,

    n_jobs=-1
)

selector.fit(
    X,
    y
)


# ============================================================
# 9. FIND SELECTED PCs
# ============================================================

selected_mask = selector.get_support()

selected_indices = np.where(
    selected_mask
)[0]

# Convert Python indices into human-readable PC numbers
selected_pcs = selected_indices + 1


print("\nSelected PCs:")
print(selected_pcs)

print(
    f"\nNumber of selected PCs: "
    f"{len(selected_pcs)}"
)


# ============================================================
# 10. FINAL REGRESSION MODEL
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


print("\nFinal regression model")

print(
    f"Intercept: "
    f"{final_model.intercept_:.4f}"
)

print("\nRegression coefficients:")

for pc, coefficient in zip(
    selected_pcs,
    final_model.coef_
):

    print(
        f"PC{pc}: "
        f"{coefficient:.6f}"
    )


# ============================================================
# 11. CROSS-VALIDATED PERFORMANCE
# ============================================================

cv_mse = -cross_val_score(
    LinearRegression(),
    X_selected,
    y,
    cv=cv,
    scoring="neg_mean_squared_error"
)

cv_r2 = cross_val_score(
    LinearRegression(),
    X_selected,
    y,
    cv=cv,
    scoring="r2"
)


print("\nCross-validation")

print(
    f"Mean MSE: "
    f"{cv_mse.mean():.4f}"
)

print(
    f"Mean RMSE: "
    f"{np.sqrt(cv_mse.mean()):.4f}"
)

print(
    f"Mean R²: "
    f"{cv_r2.mean():.4f}"
)


# ============================================================
# 12. PREDICT RATINGS FOR TRAINING FACES
# ============================================================

predicted_ratings = final_model.predict(
    X_selected
)


print("\nObserved rating range:")
print(
    f"{y.min():.2f} to {y.max():.2f}"
)

print("\nPredicted rating range:")
print(
    f"{predicted_ratings.min():.2f} "
    f"to {predicted_ratings.max():.2f}"
)


# ============================================================
# 13. OBSERVED VS PREDICTED
# ============================================================

plt.figure(figsize=(6, 6))

plt.scatter(
    y,
    predicted_ratings,
    alpha=0.6
)

minimum = min(
    y.min(),
    predicted_ratings.min()
)

maximum = max(
    y.max(),
    predicted_ratings.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Observed Mean Rating")
plt.ylabel("Predicted Rating")

plt.title(
    "Observed vs Predicted Ratings"
)

plt.tight_layout()
plt.show()

# ============================================================
# 14. GENERATE SYNTHETIC FACES
# ============================================================

# Regression weights for the selected PCs
w = final_model.coef_

# Regression intercept
delta = final_model.intercept_

# Squared length of the weight vector
w_norm_squared = np.sum(w ** 2)

print("\nSynthetic face generation")
print(f"||w||^2 = {w_norm_squared:.10f}")
print(f"Intercept = {delta:.4f}")


# Ratings requested by the assignment
target_ratings = np.arange(
    0.5,
    5.5 + 0.5,
    0.5
)

synthetic_faces = []


for target_rating in target_ratings:

    # --------------------------------------------------------
    # Equation 2.27
    #
    # alpha = (a0 - delta) / ||w||^2
    # --------------------------------------------------------

    alpha = (
        target_rating - delta
    ) / w_norm_squared


    # --------------------------------------------------------
    # Equation 2.28
    #
    # Synthetic point in SELECTED PC space
    # --------------------------------------------------------

    synthetic_selected_scores = alpha * w


    # --------------------------------------------------------
    # Put selected scores back into the full PCA space
    #
    # All non-selected PCs are set to zero.
    # --------------------------------------------------------

    synthetic_pca_scores = np.zeros(
        pca.n_components_
    )

    synthetic_pca_scores[
        selected_indices
    ] = synthetic_selected_scores


    # --------------------------------------------------------
    # Transform from PCA space back to pixel space
    #
    # PCA components describe deviations from average face,
    # so add the average face back afterwards.
    # --------------------------------------------------------

    synthetic_centered_face = (
        synthetic_pca_scores
        @ pca.components_
    )

    synthetic_face = (
        mean_face
        + synthetic_centered_face
    )


    synthetic_faces.append(
        synthetic_face
    )

    print(
        f"Target {target_rating:.1f}: "
        f"alpha = {alpha:.2f}"
    )
    


# Convert to numpy array
synthetic_faces = np.stack(
    synthetic_faces
)

# ============================================================
# 15. DISPLAY SYNTHETIC FACES
# ============================================================

fig, axes = plt.subplots(
    1,
    len(target_ratings),
    figsize=(22, 4)
)


for ax, face, rating in zip(
    axes,
    synthetic_faces,
    target_ratings
):

    # Clip only for displaying the image.
    # The underlying synthetic data are left unchanged.
    display_face = np.clip(
        face,
        0,
        255
    )

    ax.imshow(
        display_face.reshape(200, 200),
        cmap="gray",
        vmin=0,
        vmax=255
    )

    ax.set_title(
        f"{rating:.1f}"
    )

    ax.axis("off")


fig.suptitle(
    "Synthetic Faces by Predicted Rating",
    fontsize=16
)

plt.tight_layout()
plt.show()


# ============================================================
# 16. STEP 6 - SYNTHETIC FACES WITHIN PREDICTED RANGE
# ============================================================

min_predicted = predicted_ratings.min()
max_predicted = predicted_ratings.max()

print(
    f"\nPredicted training range: "
    f"{min_predicted:.2f} to {max_predicted:.2f}"
)


# Generate 11 equally spaced ratings within the range
safe_target_ratings = np.linspace(
    min_predicted,
    max_predicted,
    11
)

safe_synthetic_faces = []


for target_rating in safe_target_ratings:

    # Equation 2.27
    alpha = (
        target_rating - delta
    ) / w_norm_squared

    # Equation 2.28
    synthetic_selected_scores = (
        alpha * w
    )

    check_rating = (
    synthetic_selected_scores @ w
    + delta
    )   

    print(
        f"Target = {target_rating:.1f}, "
        f"model prediction = {check_rating:.3f}"
    )

    # Put selected PC scores into full PCA space
    synthetic_pca_scores = np.zeros(
        pca.n_components_
    )

    synthetic_pca_scores[
        selected_indices
    ] = synthetic_selected_scores

    # Transform back to pixel space
    synthetic_centered_face = (
        synthetic_pca_scores
        @ pca.components_
    )

    synthetic_face = (
        mean_face
        + synthetic_centered_face
    )

    safe_synthetic_faces.append(
        synthetic_face
    )


safe_synthetic_faces = np.stack(
    safe_synthetic_faces
)


# ============================================================
# DISPLAY SAFE SYNTHETIC CONTINUUM
# ============================================================

fig, axes = plt.subplots(
    1,
    11,
    figsize=(22, 4)
)


for ax, face, rating in zip(
    axes,
    safe_synthetic_faces,
    safe_target_ratings
):

    ax.imshow(
        np.clip(
            face.reshape(200, 200),
            0,
            255
        ),
        cmap="gray",
        vmin=0,
        vmax=255
    )

    ax.set_title(
        f"{rating:.2f}"
    )

    ax.axis("off")


fig.suptitle(
    "Synthetic Faces Within Observed Prediction Range",
    fontsize=16
)

plt.tight_layout()
plt.show()