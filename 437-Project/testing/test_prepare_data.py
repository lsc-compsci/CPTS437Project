import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from prepare_data import clean_dataset, split_dataset, remove_empty_files, is_image_file
from PIL import Image

# Robust test cases for parser (prepare_data.py)

def test_clean_dataset(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    # Add test files
    valid_file = source_dir / "1.jpg"  # Valid file
    empty_file = source_dir / "empty.txt"  # Empty file
    invalid_file = source_dir / "corrupt.png"  # Corrupt file

    # Create valid image
    image = Image.new("RGB", (100, 100), color="red")
    image.save(valid_file)

    # Create other files
    empty_file.write_text("")  # Empty file
    invalid_file.write_text("Not a real image")  # Corrupt content

    # Debug: Check before cleaning
    print("Before cleaning:", [file.name for file in source_dir.iterdir()])

    # Run clean_dataset
    clean_dataset(str(source_dir))

    # Debug: Check after cleaning
    print("After cleaning:", [file.name for file in source_dir.iterdir()])

    # Assertions
    remaining_files = [file.name for file in source_dir.iterdir()]
    assert "1.jpg" in remaining_files, "Valid image was incorrectly removed"
    assert "empty.txt" not in remaining_files, "Empty file was not removed"
    assert "corrupt.png" not in remaining_files, "Invalid file was not removed"

def test_clean_dataset_empty_dir(tmp_path):
    source_dir = tmp_path / "empty_source"
    source_dir.mkdir()

    clean_dataset(str(source_dir))

    # Assertions: The directory should remain empty
    assert os.listdir(source_dir) == [], "Empty directory should remain empty"


def test_split_dataset(tmp_path):
    # Create temporary directories
    source_dir = tmp_path / "source"
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"

    source_dir.mkdir()
    train_dir.mkdir()
    validation_dir.mkdir()

    # Create a "class" directory inside source_dir
    class_dir = source_dir / "class1"
    class_dir.mkdir()

    # Add files to the "class1" directory
    for i in range(10):
        (class_dir / f"image_{i}.jpg").write_text("test data")

    # Debug: Check contents of source_dir and class_dir before splitting
    print("Source directory before splitting:", os.listdir(source_dir))
    print("Class directory before splitting:", os.listdir(class_dir))

    # Call the function
    split_dataset(
        source_dir=str(source_dir),
        train_dir=str(train_dir),
        validation_dir=str(validation_dir),
        split_size=0.2
    )

    # Debug: Check contents of train_dir and validation_dir after splitting
    train_class_dir = train_dir / "class1"
    validation_class_dir = validation_dir / "class1"

    print("Train directory:", os.listdir(train_class_dir))
    print("Validation directory:", os.listdir(validation_class_dir))

    # Assertions: Check the split ratio
    assert len(os.listdir(train_class_dir)) == 8  # 80%
    assert len(os.listdir(validation_class_dir)) == 2  # 20%

def test_remove_empty_files(tmp_path):
    # Create a temporary directory
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    # Create zero-byte and non-zero-byte files
    empty_file = source_dir / "empty_file.txt"
    empty_file.touch()  # Create an empty file

    non_empty_file = source_dir / "non_empty_file.txt"
    non_empty_file.write_text("This is not empty")  # Write some content

    # Create a nested subdirectory with an empty file
    sub_dir = source_dir / "subdir"
    sub_dir.mkdir()
    empty_file_in_subdir = sub_dir / "empty_in_subdir.txt"
    empty_file_in_subdir.touch()

    # Run the function
    remove_empty_files(str(source_dir))

    # Assertions
    assert not empty_file.exists(), "Empty file was not removed"
    assert non_empty_file.exists(), "Non-empty file was removed"
    assert not empty_file_in_subdir.exists(), "Empty file in subdirectory was not removed"

def test_is_image_file(tmp_path):
    # Create a valid image file
    valid_image = tmp_path / "valid_image.jpg"
    image = Image.new("RGB", (100, 100), color="red")
    image.save(valid_image)

    # Create an invalid image file
    invalid_image = tmp_path / "invalid_image.jpg"
    invalid_image.write_text("This is not an image")  # Write non-image content

    # Create a non-image file
    non_image_file = tmp_path / "not_an_image.txt"
    non_image_file.write_text("This is a text file")

    # Run the function on each file
    assert is_image_file(str(valid_image)) is True, "Valid image was not recognized"
    assert valid_image.exists(), "Valid image was removed"

    assert is_image_file(str(invalid_image)) is False, "Invalid image was not detected"
    assert not invalid_image.exists(), "Invalid image was not removed"

    assert is_image_file(str(non_image_file)) is False, "Non-image file was not detected"
    assert not non_image_file.exists(), "Non-image file was not removed"