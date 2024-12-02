import os
import shutil
import random

# Paths
original_dataset_dir_cats = 'all_data/Cat'   # Path to original cat images
original_dataset_dir_dogs = 'all_data/Dog'   # Path to original dog images
base_dir = 'data'                            # Base directory for train and validation folders

# Directories for train and validation data
train_cats_dir = os.path.join(base_dir, 'train/cats')
train_dogs_dir = os.path.join(base_dir, 'train/dogs')
validation_cats_dir = os.path.join(base_dir, 'validation/cats')
validation_dogs_dir = os.path.join(base_dir, 'validation/dogs')

# Ensure directories exist (this is just a safety check)
directories = [
    train_cats_dir, train_dogs_dir,
    validation_cats_dir, validation_dogs_dir
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Function to split data and copy files
def split_and_copy_files(src_dir, train_dir, validation_dir, split_size=0.8):
    filenames = os.listdir(src_dir)
    filenames = [f for f in filenames if os.path.isfile(os.path.join(src_dir, f))]  # Filter out any non-file entries

    # Shuffle filenames
    random.shuffle(filenames)

    # Split filenames
    split_index = int(len(filenames) * split_size)
    train_filenames = filenames[:split_index]
    validation_filenames = filenames[split_index:]

    # Copy files to training directory
    for fname in train_filenames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(train_dir, fname)
        shutil.copyfile(src, dst)

    # Copy files to validation directory
    for fname in validation_filenames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(validation_dir, fname)
        shutil.copyfile(src, dst)

    print(f"Copied {len(train_filenames)} files to {train_dir}")
    print(f"Copied {len(validation_filenames)} files to {validation_dir}")

# Split and copy cat images
split_and_copy_files(
    src_dir=original_dataset_dir_cats,
    train_dir=train_cats_dir,
    validation_dir=validation_cats_dir,
    split_size=0.8  # 80% for training, 20% for validation
)

# Split and copy dog images
split_and_copy_files(
    src_dir=original_dataset_dir_dogs,
    train_dir=train_dogs_dir,
    validation_dir=validation_dogs_dir,
    split_size=0.8
)

print('Data preparation completed.')
