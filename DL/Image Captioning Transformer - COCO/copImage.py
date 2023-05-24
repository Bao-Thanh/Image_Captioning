import shutil
import os

def copy_images(source_folder, destination_folder, num_images):
    os.makedirs(destination_folder, exist_ok=True)
    
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    image_files.sort()
    
    num_images = min(num_images, len(image_files))
    
    copied_files = []
    for i in range(num_images):
        source_file = os.path.join(source_folder, image_files[i])
        destination_file = os.path.join(destination_folder, image_files[i])
        shutil.copy2(source_file, destination_file)
        copied_files.append(destination_file)
    
    return copied_files


source_folder = 'dataset\\val2014'
destination_folder = 'img'
num_images = 1000

copied_files = copy_images(source_folder, destination_folder, num_images)
print(f"{len(copied_files)} image files have been copied to the destination folder.")
