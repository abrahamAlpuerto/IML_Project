from PIL import Image
from pillow_heif import register_heif_opener
import os

register_heif_opener()
folder_path = "j/jessy"
heic_files = []
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path,filename)
    heic_files.append(file_path)


for photo in heic_files:

    temp_img = Image.open(photo)
    jpg_photo = photo.replace('.HEIC', '.jpg')
    temp_img.save(jpg_photo)