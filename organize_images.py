import os
import shutil
from pathlib import Path

# Folder containing raw images (e.g., cat.0.jpg, dog.1.jpg)
base_dir = Path("data/train")

# Destination folders
cat_dir = base_dir / "cats"
dog_dir = base_dir / "dogs"

# Make sure the destination folders exist
cat_dir.mkdir(parents=True, exist_ok=True)
dog_dir.mkdir(parents=True, exist_ok=True)

# Move cat images
for img_file in base_dir.glob("cat.*.jpg"):
    print(f"Moving {img_file.name} to cats/")
    shutil.move(str(img_file), str(cat_dir / img_file.name))

# Move dog images
for img_file in base_dir.glob("dog.*.jpg"):
    print(f"Moving {img_file.name} to dogs/")
    shutil.move(str(img_file), str(dog_dir / img_file.name))

print("✅ Done moving cat and dog images.")
