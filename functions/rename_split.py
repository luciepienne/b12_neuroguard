from sklearn.model_selection import train_test_split
import os
import shutil

# Define function to copy files and rename them
def copy_and_rename_files(source_path, dest_path, files, prefix, counter):
    for file in files:
        # Generate new filename
        new_filename = f"{prefix}{counter:05}.jpg"
        # Increment counter
        counter += 1
        # Copy and rename file
        shutil.copy(os.path.join(source_path, file), os.path.join(dest_path, new_filename))
    return counter

# Filter out files ending with 'Zone.Identifier'
def filter_files(files):
    return [file for file in files if not file.endswith('Zone.Identifier')]