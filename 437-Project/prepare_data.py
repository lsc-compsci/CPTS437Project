import os
import shutil
import random
from PIL import Image

def remove_empty_files(directory):
    """
    Removes all zero-byte files in the specified directory and its subdirectories.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getsize(file_path) == 0:
                print(f"Removing empty file: {file_path}")
                os.remove(file_path)

def is_image_file(filepath):
    """
    Checks whether a file is a valid image.
    """
    try:
        with Image.open(filepath) as img:
            img.verify()  # Verify that it is, in fact, an image
        return True
    except (IOError, SyntaxError):
        print(f"Invalid image file detected and removed: {filepath}")
        os.remove(filepath)
        return False

def clean_dataset(source_dir):
    """
    Cleans the dataset by removing zero-byte and invalid image files.
    """
    print("Starting dataset cleanup...")
    remove_empty_files(source_dir)
    
    print("Verifying image files...")
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            is_image_file(file_path)
    print("Dataset cleanup completed.")

def split_dataset(source_dir, train_dir, validation_dir, split_size=0.2):
    """
    Splits the dataset into training and validation sets.
    
    Parameters:
    - source_dir: Directory containing the original data.
    - train_dir: Directory to store training data.
    - validation_dir: Directory to store validation data.
    - split_size: Proportion of data to be used for validation.
    """
    print("Starting dataset split...")
    # Ensure destination directories exist
    for directory in [train_dir, validation_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Iterate over each class in the source directory
    for class_name in os.listdir(source_dir):
        class_source = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_source):
            continue  # Skip non-directory files
        
        # Define destination paths
        train_class_dir = os.path.join(train_dir, class_name)
        validation_class_dir = os.path.join(validation_dir, class_name)
        
        for directory in [train_class_dir, validation_class_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
        
        # List all image files in the class directory
        all_files = [f for f in os.listdir(class_source) if os.path.isfile(os.path.join(class_source, f))]
        random.shuffle(all_files)  # Shuffle to ensure random distribution
        
        # Calculate split index
        split_index = int(len(all_files) * (1 - split_size))
        train_files = all_files[:split_index]
        validation_files = all_files[split_index:]
        
        # Copy files to training directory
        for file in train_files:
            src_file = os.path.join(class_source, file)
            dst_file = os.path.join(train_class_dir, file)
            shutil.copy2(src_file, dst_file)
        
        # Copy files to validation directory
        for file in validation_files:
            src_file = os.path.join(class_source, file)
            dst_file = os.path.join(validation_class_dir, file)
            shutil.copy2(src_file, dst_file)
        
        print(f"Split {class_name}: {len(train_files)} training and {len(validation_files)} validation images.")
    
    print("Dataset split completed.")

def main():
    # Define source and destination directories
    source_dataset_dir = 'all_data'  # Original dataset directory
    train_dir = 'data/train'
    validation_dir = 'data/validation'
    
    # Step 1: Clean the dataset by removing empty and invalid image files
    clean_dataset(source_dataset_dir)
    
    # Step 2: Split the dataset into training and validation sets
    split_dataset(
        source_dir=source_dataset_dir,
        train_dir=train_dir,
        validation_dir=validation_dir,
        split_size=0.2  # 80% training and 20% validation
    )
    
    print("Data preparation is complete.")

if __name__ == "__main__":
    main()
