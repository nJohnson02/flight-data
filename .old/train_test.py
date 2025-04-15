import os
import shutil
import random
from pathlib import Path

# Set paths
base_dir = Path(__file__).resolve().parent
classified_dir = base_dir / "manual_classification"
good_dir = classified_dir / "good"
bad_dir = classified_dir / "bad"

train_dir = base_dir / "train"
test_dir = base_dir / "test"

# Create output folders
for split_dir in [train_dir, test_dir]:
    (split_dir / "good").mkdir(parents=True, exist_ok=True)
    (split_dir / "bad").mkdir(parents=True, exist_ok=True)

# Collect and shuffle files
good_files = list(good_dir.glob("*.csv"))
bad_files = list(bad_dir.glob("*.csv"))
random.seed(69)
random.shuffle(good_files)
random.shuffle(bad_files)

# Split
split_ratio = 0.8
good_split = int(len(good_files) * split_ratio)
bad_split = int(len(bad_files) * split_ratio)

train_good = good_files[:good_split]
test_good = good_files[good_split:]
train_bad = bad_files[:bad_split]
test_bad = bad_files[bad_split:]

# Copy files to train/test directories
def copy_files(files, target_dir):
    for file in files:
        shutil.copy(file, target_dir / file.name)

copy_files(train_good, train_dir / "good")
copy_files(test_good, test_dir / "good")
copy_files(train_bad, train_dir / "bad")
copy_files(test_bad, test_dir / "bad")

print(f"Split complete:")
print(f"  Train set → {len(train_good)} good, {len(train_bad)} bad")
print(f"  Test set  → {len(test_good)} good, {len(test_bad)} bad")
