import os

# === Set your folder paths here ===
reference_folder = "src/My Team-images"  # Folder with the correct filenames
target_folder = "src/My Team-npy"        # Folder to clean up

# === Get filenames without extensions from the reference folder ===
reference_files = os.listdir(reference_folder)
reference_names = {os.path.splitext(f)[0] for f in reference_files}

# === Iterate over files in the target folder and delete unmatched files ===
for file in os.listdir(target_folder):
    file_path = os.path.join(target_folder, file)
    
    if os.path.isfile(file_path):
        name_without_ext = os.path.splitext(file)[0]
        if name_without_ext not in reference_names:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print(f"Kept: {file_path}")
