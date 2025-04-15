import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

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
    vert_accel = np.diff(alt_rate, prepend=alt_rate[0])  # ft/min²
    
    return np.concatenate([alts, alt_rate, vert_accel])


def load_dataset(split_dir):
    X, y = [], []
    for label, folder in [(0, "good"), (1, "bad")]:
        class_dir = split_dir / folder
        for file in class_dir.glob("*.csv"):
            vec = load_and_resample(file, n_points)
            if vec is not None:
                X.append(vec)
                y.append(label)
    return np.array(X), np.array(y)


# --- Main Execution ---
base_dir = Path(__file__).resolve().parent
train_dir = base_dir / "train"
test_dir = base_dir / "test"

print("Loading training data...")
X_train, y_train = load_dataset(train_dir)
print(f"Loaded {len(X_train)} training samples.")

print("Loading test data...")
X_test, y_test = load_dataset(test_dir)
print(f"Loaded {len(X_test)} test samples.")

# Train classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
