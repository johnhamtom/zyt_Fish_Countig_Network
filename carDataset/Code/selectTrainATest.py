# Extract images based on the given train.txt and test .txt
import os
import shutil

def select_and_copy_images(txt_file, source_folder, destination_folder):
    # Create destination folder (if it doesn't exist)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Read the image name in the txt file
    with open(txt_file, 'r') as file:
        image_names = file.read().splitlines()

    # Select and copy pictures from the source folder to the destination folder
    for image_name in image_names:
        source_path = os.path.join(source_folder, image_name+'.png')
        print(source_path)
        destination_path = os.path.join(destination_folder, image_name+'.png')
        if os.path.exists(source_path):
            shutil.copyfile(source_path, destination_path)
            print("The picture has been copied: " + image_name)
        else:
            print("Picture not found:" + image_name)

# 使用示例
file=['train','test']
for i in file:
    txt_file = os.path.join(r'/home/chuanzhi/mnt_3T/zyt/CARPKandPUCPR/CARPK_devkit/data/ImageSets',i+'.txt')
    source_folder = os.path.join(r'/home/chuanzhi/mnt_3T/zyt/CARPKandPUCPR/CARPK_devkit/data/Images')
    destination_folder = os.path.join(r'/home/chuanzhi/mnt_3T/zyt/CARPKandPUCPR/CARPK_devkit/data/experiment/DA-origin', i)
    select_and_copy_images(txt_file, source_folder, destination_folder)
