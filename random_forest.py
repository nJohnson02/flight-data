import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# --- Config ---
n_points = 60  # use the final 60 time steps of each approach

def load_and_resample(csv_path, n_points):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df = df[df["AltGPS"].notna() & df["VSpd"].notna()]

    if len(df) < n_points:
        return None

    alts = df["AltGPS"].astype(float).values[-n_points:]
    alt_rate = np.diff(alts, prepend=alts[0]) * 60  # ft/min

    return np.concatenate([alts, alt_rate])

def load_dataset(classified_dir):
    X, y = [], []
    for label, folder in [(0, "good"), (1, "bad")]:
        class_dir = classified_dir / folder
        for file in class_dir.glob("*.csv"):
            vec = load_and_resample(file, n_points)
            if vec is not None:
                X.append(vec)
                y.append(label)
    return np.array(X), np.array(y)

# --- Main Execution ---
base_dir = Path(__file__).resolve().parent
classified_dir = base_dir / "manual_classification"

print("Loading all labeled data...")
X, y = load_dataset(classified_dir)
print(f"Loaded {len(X)} total samples.")

# Run k-fold cross-validation
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)

print("\nCross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())

# Train on full dataset to visualize one tree
clf.fit(X, y)

# Plot the first tree in the forest
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(clf.estimators_[0], filled=True, max_depth=3, feature_names=[f"f{i}" for i in range(X.shape[1])])
plt.title("First Tree in Random Forest (max depth = 3)")
plt.show()
