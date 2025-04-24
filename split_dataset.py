import os
import random
import shutil
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

# Paths
base_dir = Path("data/train")
train_out = Path("data/train")
val_out = Path("data/val")

# Subfolders (cats and dogs)
classes = ['cats', 'dogs']

# Percentage for validation set
val_split = 0.2

for cls in classes:
    src_dir = base_dir / cls
    all_files = list(src_dir.glob("*.jpg"))
    random.shuffle(all_files)

    val_count = int(len(all_files) * val_split)
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]

    # Make destination folders
    (val_out / cls).mkdir(parents=True, exist_ok=True)
    (train_out / cls).mkdir(parents=True, exist_ok=True)

    # Move files
    for f in val_files:
        shutil.move(str(f), str(val_out / cls / f.name))

    for f in train_files:
        shutil.move(str(f), str(train_out / cls / f.name))

print("✅ Done splitting into train/val folders.")
