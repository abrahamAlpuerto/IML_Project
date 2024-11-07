import os
import shutil

# Specify the source and destination folders
source_folder = "j/jessy"
destination_folder = "pics"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Loop through all files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file ends with .jpg
    if filename.lower().endswith(".jpg"):
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        
        # Move the file
        shutil.move(source_path, destination_path)
        print(f"Moved {filename} to {destination_folder}")