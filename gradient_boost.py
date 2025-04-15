import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier, plot_tree
import matplotlib.pyplot as plt

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
classified_dir = base_dir / "data"

print("Loading all labeled data...")
X, y = load_dataset(classified_dir)
print(f"Loaded {len(X)} total samples.")

# Run k-fold cross-validation
clf = XGBClassifier(n_estimators=100, max_depth=5, eval_metric='logloss')
scores = cross_val_score(clf, X, y, cv=5)

print("\nCross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())

# Train on a train/test split to show confusion matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Retrain on full data to visualize one tree
clf.fit(X, y)

plt.figure(figsize=(20, 10))
plot_tree(clf, num_trees=0, rankdir='LR')
plt.title("First Tree in Gradient Boosting Model")
plt.show()
