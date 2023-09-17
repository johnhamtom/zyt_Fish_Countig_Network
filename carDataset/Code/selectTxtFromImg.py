# Extract images according to a certain interval

import os
import shutil

source_path =  r"/home/chuanzhi/mnt_3T/zyt/BLfishData/fish/New/val"  # Image path
destination_path = r"/home/chuanzhi/mnt_3T/zyt/BLfishData/fish/New/valtxt"
files = os.listdir(source_path)
for file in files:
    img_path = os.path.join(source_path,file)
    txt_path = img_path.replace('.JPG','.txt')
    shutil.copyfile(txt_path, os.path.join(destination_path,os.path.basename(txt_path)))
