import os

def rename_files_in_series(folder1, folder2, target_folder):
    """
    Renames and moves common .npy files from folder1 and folder2 to target_folder in a numbered series.

    Args:
    folder1 (str): Path to the first folder (e.g., wireframe).
    folder2 (str): Path to the second folder (e.g., ground truth).
    target_folder (str): Path to the folder where renamed files will be saved.
    """
    # Get the list of files in both folders
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # Find the common files between both folders
    common_files = files1.intersection(files2)

    if not common_files:
        print("No common files found in both folders.")
        return

    # Ensure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Rename the common files in series order
    for i, file_name in enumerate(common_files, start=1):
        # Get full paths of the file in both folders
        file1_path = os.path.join(folder1, file_name)
        file2_path = os.path.join(folder2, file_name)

        # Extract file extension (assuming files have extensions)
        _, ext = os.path.splitext(file_name)

        if ext != ".npy":
            continue  # Skip files that are not .npy format

        # Rename files and move them to the target folder with series number
        new_name_1 = f"{i}_1{ext}"
        new_name_2 = f"{i}_2{ext}"

        new_file1_path = os.path.join(target_folder, new_name_1)
        new_file2_path = os.path.join(target_folder, new_name_2)

        # Move and rename the files
        os.rename(file1_path, new_file1_path)
        os.rename(file2_path, new_file2_path)

        print(f"Renamed and moved: {file_name} to {new_name_1} and {new_name_2}")

# Example usage: Wireframe and Ground Truth folders with .npy files
wireframe_folder = "/path/to/wireframe"
ground_truth_folder = "/path/to/ground_truth"
output_folder = "/path/to/output"

# Rename and move common .npy files from wireframe and ground_truth to output folder
rename_files_in_series(wireframe_folder, ground_truth_folder, output_folder)
