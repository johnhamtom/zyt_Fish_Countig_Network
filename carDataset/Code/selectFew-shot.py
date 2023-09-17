# Extract images according to a certain interval

import os
import shutil

def extract_images(source_folder, destination_folder, interval):
    # Create destination folder (if it doesn't exist)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get all picture files in the source folder and sort by file name
    image_files = sorted([file for file in os.listdir(source_folder) if file.endswith('.jpg') or file.endswith('.png')])

    # Extract pictures at regular intervals starting from the first picture and copy to the destination folder
    for i in range(0, len(image_files), interval):
        image_name = image_files[i]
        source_path = os.path.join(source_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)
        shutil.copyfile(source_path, destination_path)
        '''
        # copy .npy
        source_path_npy = os.path.join(source_folder, image_name.replace('.jpg','.npy'))
        destination_path_npy = os.path.join(destination_folder, image_name.replace('.jpg','.npy'))
        shutil.copyfile(source_path_npy, destination_path_npy)
        
        # copy .mat
        path = r"/home/chuanzhi/mnt_3T/zyt/people/UCF-QNRF_ECCV18 (2)/UCF-QNRF_ECCV18/val"
        imageName = image_name.split('.')
        mat_path = os.path.join(path,imageName[0]+'_ann'+'.mat')
        destination_path_mat = os.path.join(destination_folder+'mat',image_name.replace('.jpg','.mat'))
        print(mat_path)
        
        shutil.copyfile(mat_path, destination_path_mat)
        print(f"The picture has been copied: {image_name}")
	'''
# 使用示例
source_folder = r'/home/chuanzhi/mnt_3T/zyt/people/UCF-Train-Val-Test/val'  # The path of the folder where the source image is located
destination_folder = r'/home/chuanzhi/mnt_3T/zyt/people/select-UCF/val'  # The destination folder path where the pictures are stored after extraction
interval = 10  # Extraction interval

extract_images(source_folder, destination_folder, interval)
